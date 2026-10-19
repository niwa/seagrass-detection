"""Module focused on downloading and processing Sentinel-2 data."""

import pathlib
import zoneinfo
import utils
import datetime
import requests
import odc.stac
import planetary_computer
import leafmap
import pandas
import numpy
import xarray
from rioxarray.exceptions import NoDataInBounds
import dotenv
import os
import time


CATALOGUE_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

SATELLITE_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
DATE_FORMAT_SITE_SURVEY = "%m/%d/%Y"
DATE_FORMAT_YYYYMMDD = "%Y-%m-%d"
TIDE_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

TIDE_API_STUB = "https://api.niwa.co.nz/tides/data"
TIDE_API_RETRY_DELAY_SECONDS = 1
LOW_TIDE_DELTA = 2  # in hrs

S2_RESOLUTION = 10
BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "B11",
    "B12",
    "B8A",
    "SCL",
]
RGB_BANDS = ["B04", "B03", "B02"]
SCL_TO_IGNORE = [
    0,  # NO_DATA
    1,  # SATURATED_OR_DEFECTIVE
    3,  # CLOUD_SHADOWS
    8,  # CLOUD_MEDIUM_PROBABILITY
    9,  # CLOUD_HIGH_PROBABILITY
    10,  # THIN_CIRRUS
    11, # SNOW
]  

HARMONIZE_DATE = "2022-01-25"
BAND_OFFSET_POST_2022_01_25 = 1000


def empty_satellite_dataset():
    """Return an empty satellite dataset with the expected bands and dimensions."""
    dimensions = ("time", "y", "x")
    empty_data = numpy.empty((0, 0, 0), dtype="uint16")
    return xarray.Dataset(
        {band: (dimensions, empty_data.copy()) for band in BANDS},
        coords={"time": [], "y": [], "x": []},
    )


def stac_search_with_retry(max_retries: int = 5, initial_delay_seconds: float = 5.0, **stac_search_kwargs):
    """Call leafmap.stac_search with retries and exponential backoff, to handle
    transient STAC catalogue errors (e.g. 502 Bad Gateway from the Planetary
    Computer API)."""

    delay_seconds = initial_delay_seconds
    for attempt in range(1, max_retries + 1):
        try:
            return leafmap.stac_search(**stac_search_kwargs)
        except Exception as error:
            if attempt == max_retries:
                raise
            print(
                f"\tSTAC search failed (attempt {attempt}/{max_retries}): {error}. "
                f"Retrying in {delay_seconds:.0f}s..."
            )
            time.sleep(delay_seconds)
            delay_seconds *= 2


def get_satellite_date_range(site_name: str, date_file: pathlib.Path,
                             search_days: int):
    """Read in the data file and return a date range given the
    number of search days specified. The date range is centered
    around the survey date."""
    dates = pandas.read_csv(date_file)
    survey_date = dates[dates["site"] == site_name]["date"].iloc[0]
    survey_date = datetime.datetime.strptime(
        survey_date, DATE_FORMAT_SITE_SURVEY
    ).date()
    search_days = datetime.timedelta(days=int(search_days / 2))
    date_range = (
        f"{(survey_date - search_days).strftime(DATE_FORMAT_YYYYMMDD)}"
        "/"
        f"{(survey_date + search_days).strftime(DATE_FORMAT_YYYYMMDD)}"
    )
    return date_range


def _get_tide_api_filtered_images_near_date(
    site_name,
    geometry,
    date_file,
    low_tide_search_days: int,
    low_tide_delta_hrs: int = None,
    low_tide_delta_mins: int = None,
):
    """Return satellite images near a survey date selected by tide or SCL threshold."""
    date_range = get_satellite_date_range(
        site_name, date_file, search_days=low_tide_search_days
    )
    print(f"\tSatellite date range {date_range}")
    geometry_WSG = geometry.buffer(S2_RESOLUTION).to_crs(
        utils.CRS_WSG
    )  # ensure includes pixles on edge

    search_collection = stac_search_with_retry(
        url=CATALOGUE_URL,
        max_items=200,
        collections=[COLLECTION],
        bbox=geometry_WSG.total_bounds,
        datetime=date_range,
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        get_collection=True,
    )

    items = search_collection.items

    # Optional solar-angle screen here: use item.properties["s2:mean_solar_zenith"] (elevation = 90 - zenith) and ["s2:mean_solar_azimuth"] before tide checks or loading assets.

    all_tide_n = len(items)
    centroid_nzgd49 = geometry.to_crs(utils.CRS_NZGD49).geometry.union_all().centroid
    latitude_nzgd49, longitude_nzgd49 = float(centroid_nzgd49.y), float(centroid_nzgd49.x)
    low_tide = [
        _check_low_tide_from_tide_api(
            item,
            lat=latitude_nzgd49,
            lon=longitude_nzgd49,
            low_tide_delta_hrs=low_tide_delta_hrs,
            low_tide_delta_mins=low_tide_delta_mins,
        )
        for item in items
    ]
    items = [item for item, is_low_tide in zip(items, low_tide) if is_low_tide]
    print(f"\tTide API-filtered tiles: {len(items)} from {all_tide_n} STAC items")

    if len(items) == 0:
        return empty_satellite_dataset()
    
    # Keep all near lowtide
    data = odc.stac.load(
        items,
        bbox=geometry_WSG.total_bounds,
        bands=BANDS,
        chunks={},
        groupby="solar_day",
        resolution=S2_RESOLUTION,
        dtype="uint16",
        nodata=0,
        patch_url=planetary_computer.sign,
    )
    data = data.rio.clip(
        geometry.to_crs(data.rio.crs).geometry, all_touched=True, drop=True
    )
    data.load()
    data = harmonize_post_2022(data)

    print(
        f"\tTide API-filtered tiles: {len(data['time'])} from "
        f"{all_tide_n} STAC items"
    )

    return data


def _get_mndwi_thresholded_images_near_date(
    site_name,
    geometry,
    date_file,
    low_tide_search_days: int,
    mndwi_threshold: float = None,
):
    """Return satellite images near a survey date selected by tide or SCL threshold."""
    date_range = get_satellite_date_range(
        site_name, date_file, search_days=low_tide_search_days
    )
    print(f"\tSatellite date range {date_range}")
    geometry_WSG = geometry.buffer(S2_RESOLUTION).to_crs(
        utils.CRS_WSG
    )  # ensure includes pixles on edge

    search_collection = stac_search_with_retry(
        url=CATALOGUE_URL,
        max_items=200,
        collections=[COLLECTION],
        bbox=geometry_WSG.total_bounds,
        datetime=date_range,
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        get_collection=True,
    )

    items = search_collection.items

    # Optional solar-angle screen here: use item.properties["s2:mean_solar_zenith"] (elevation = 90 - zenith) and ["s2:mean_solar_azimuth"] before tide checks or loading assets.

    all_tide_n = len(items)

    if len(items) == 0:
        return empty_satellite_dataset()
    
    # Keep all near lowtide
    data = odc.stac.load(
        items,
        bbox=geometry_WSG.total_bounds,
        bands=BANDS,
        chunks={},
        groupby="solar_day",
        resolution=S2_RESOLUTION,
        dtype="uint16",
        nodata=0,
        patch_url=planetary_computer.sign,
    )
    data = data.rio.clip(
        geometry.to_crs(data.rio.crs).geometry, all_touched=True, drop=True
    )
    data.load()
    data = harmonize_post_2022(data)


    data = _threshold_low_tide_dates(data, mndwi_threshold)
    print(
        f"\tMNDWI thresholded tiles: {len(data['time'])} from "
        f"{all_tide_n} STAC items"
    )

    return data


def _get_images_near_date(
    site_name,
    geometry,
    date_file,
    low_tide_search_days: int,
):
    """Return satellite images near a survey date selected by tide or SCL threshold."""
    date_range = get_satellite_date_range(
        site_name, date_file, search_days=low_tide_search_days
    )
    print(f"\tSatellite date range {date_range}")
    geometry_WSG = geometry.buffer(S2_RESOLUTION).to_crs(
        utils.CRS_WSG
    )  # ensure includes pixles on edge

    search_collection = stac_search_with_retry(
        url=CATALOGUE_URL,
        max_items=200,
        collections=[COLLECTION],
        bbox=geometry_WSG.total_bounds,
        datetime=date_range,
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        get_collection=True,
    )

    items = search_collection.items
    # Optional solar-angle screen here: use item.properties["s2:mean_solar_zenith"] (elevation = 90 - zenith) and ["s2:mean_solar_azimuth"] before tide checks or loading assets.

    if len(items) == 0:
        return empty_satellite_dataset()
    
    # Keep all near lowtide
    data = odc.stac.load(
        items,
        bbox=geometry_WSG.total_bounds,
        bands=BANDS,
        chunks={},
        groupby="solar_day",
        resolution=S2_RESOLUTION,
        dtype="uint16",
        nodata=0,
        patch_url=planetary_computer.sign,
    )
    data = data.rio.clip(
        geometry.to_crs(data.rio.crs).geometry, all_touched=True, drop=True
    )
    data.load()
    data = harmonize_post_2022(data)

    return data


def _get_images_in_year(
    geometry,
    year: int,
):
    """Return annual Sentinel-2 images filtered by a tide delta or dry threshold."""
    geometry_WSG = geometry.buffer(S2_RESOLUTION).to_crs(
        utils.CRS_WSG
    )  # ensure includes pixles on edge

    search_collection = leafmap.stac_search(
        url=CATALOGUE_URL,
        max_items=400,
        collections=[COLLECTION],
        bbox=geometry_WSG.total_bounds,
        datetime=f"{year}-01-01/{year}-12-31",
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        get_collection=True,
    )

    items = search_collection.items
    # Optional solar-angle screen here: use item.properties["s2:mean_solar_zenith"] (elevation = 90 - zenith) and ["s2:mean_solar_azimuth"] before tide checks or loading assets.    

    if len(items) == 0:
        return empty_satellite_dataset()

    # Keep only those with no cloud cover
    data = odc.stac.load(
        items,
        bbox=geometry_WSG.total_bounds,
        bands=BANDS,
        chunks={},
        groupby="solar_day",
        resolution=S2_RESOLUTION,
        dtype="uint16",
        nodata=0,
        patch_url=planetary_computer.sign,
    )
    data = data.rio.clip(
        geometry.to_crs(data.rio.crs).geometry, all_touched=True, drop=True
    )
    data.load()
    data = harmonize_post_2022(data)

    return data

def _get_mndwi_thresholded_images_in_year(
    geometry,
    year: int,
    mndwi_threshold: float = None,
):
    """Return annual Sentinel-2 images filtered by a tide delta or dry threshold."""
    geometry_WSG = geometry.buffer(S2_RESOLUTION).to_crs(
        utils.CRS_WSG
    )  # ensure includes pixles on edge

    search_collection = leafmap.stac_search(
        url=CATALOGUE_URL,
        max_items=400,
        collections=[COLLECTION],
        bbox=geometry_WSG.total_bounds,
        datetime=f"{year}-01-01/{year}-12-31",
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        get_collection=True,
    )

    items = search_collection.items

    # Optional solar-angle screen here: use item.properties["s2:mean_solar_zenith"] (elevation = 90 - zenith) and ["s2:mean_solar_azimuth"] before tide checks or loading assets.  
  
    all_tide_n = len(items)
    if len(items) == 0:
        return empty_satellite_dataset()

    # Keep all near lowtide and keep only those with no cloud cover
    data = odc.stac.load(
        items,
        bbox=geometry_WSG.total_bounds,
        bands=BANDS,
        chunks={},
        groupby="solar_day",
        resolution=S2_RESOLUTION,
        dtype="uint16",
        nodata=0,
        patch_url=planetary_computer.sign,
    )
    data = data.rio.clip(
        geometry.to_crs(data.rio.crs).geometry, all_touched=True, drop=True
    )
    data.load()
    data = harmonize_post_2022(data)
    
    data = _threshold_low_tide_dates(data, mndwi_threshold)
    print(
        f"\tmean MNDWI less than {mndwi_threshold} filtered tiles: {len(data['time'])} from "
        f"{all_tide_n} STAC items"
    )

    return data

def _get_tide_api_filtered_images_in_year(
    geometry,
    year: int,
    low_tide_delta_hrs: int = None,
    low_tide_delta_mins: int = None,
):
    """Return annual Sentinel-2 images filtered by a tide delta or dry threshold."""
    geometry_WSG = geometry.buffer(S2_RESOLUTION).to_crs(
        utils.CRS_WSG
    )  # ensure includes pixles on edge

    search_collection = leafmap.stac_search(
        url=CATALOGUE_URL,
        max_items=400,
        collections=[COLLECTION],
        bbox=geometry_WSG.total_bounds,
        datetime=f"{year}-01-01/{year}-12-31",
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        get_collection=True,
    )

    items = search_collection.items
    # Optional solar-angle screen here: use item.properties["s2:mean_solar_zenith"] (elevation = 90 - zenith) and ["s2:mean_solar_azimuth"] before tide checks or loading assets.    
    all_tide_n = len(items)
    
    centroid_nzgd49 = geometry.to_crs(utils.CRS_NZGD49).geometry.union_all().centroid
    latitude_nzgd49, longitude_nzgd49 = float(centroid_nzgd49.y), float(centroid_nzgd49.x)
    low_tide = [
        _check_low_tide_from_tide_api(
            item,
            lat=latitude_nzgd49,
            lon=longitude_nzgd49,
            low_tide_delta_hrs=low_tide_delta_hrs,
            low_tide_delta_mins=low_tide_delta_mins,
        )
        for item in items
    ]
    items = [item for item, is_low_tide in zip(items, low_tide) if is_low_tide]
    print(f"\tTide API-filtered tiles: {len(items)} from {all_tide_n} STAC items")

    if len(items) == 0:
        return empty_satellite_dataset()

    # Keep all near lowtide and keep only those with no cloud cover
    data = odc.stac.load(
        items,
        bbox=geometry_WSG.total_bounds,
        bands=BANDS,
        chunks={},
        groupby="solar_day",
        resolution=S2_RESOLUTION,
        dtype="uint16",
        nodata=0,
        patch_url=planetary_computer.sign,
    )
    data = data.rio.clip(
        geometry.to_crs(data.rio.crs).geometry, all_touched=True, drop=True
    )
    data.load()
    data = harmonize_post_2022(data)
    
    print(
        f"\tTide API-filtered tiles: {len(data['time'])} from "
        f"{all_tide_n} STAC items"
    )

    return data


def get_satellite_for_date(geometry, date_YYMMDD: str):
    """Reture satellite images near the survey date near low tide."""
    geometry_WSG = geometry.buffer(S2_RESOLUTION).to_crs(
        utils.CRS_WSG
    )  # ensure includes pixles on edge

    search_collection = stac_search_with_retry(
        url=CATALOGUE_URL,
        max_items=400,
        collections=[COLLECTION],
        bbox=geometry_WSG.total_bounds,
        datetime=f"{date_YYMMDD}",
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        get_collection=True,
    )

    items = search_collection.items

    return items


def get_low_tide_no_cloud_images_near_date(
    site_name, geometry, max_cloud_cover, date_file,
    low_tide_search_days, low_tide_delta_hrs: int = None,
    low_tide_delta_mins: int = None,
    anomaly_filter_baseline: pandas.DataFrame = None,
    n_mad: float = 3.0,
):

    data = _get_tide_api_filtered_images_near_date(
        site_name=site_name,
        geometry=geometry,
        date_file=date_file,
        low_tide_search_days=low_tide_search_days,
        low_tide_delta_hrs=low_tide_delta_hrs,
        low_tide_delta_mins=low_tide_delta_mins,
    )

    number_of_low_tide_dates = len(data["time"])

    if number_of_low_tide_dates == 0:
        return data

    data, cloud_cover_percentage, kept = filter_on_max_scl_ignore(data, max_cloud_cover, geometry)
    print(
        f"\tLow Cloud Tiles: {len(data['time'])} from the low tide tiles of "
        f"{number_of_low_tide_dates}. Cloud percentages {numpy.round(cloud_cover_percentage.data, decimals=1)} "
        f"Kept {kept.data}"
    )

    if len(data["time"]) == 0:
        return data

    if anomaly_filter_baseline is not None:
        data, deviation_per_band, kept = flag_anomalous_band_dates(
            data, baseline=anomaly_filter_baseline, n_mad=n_mad
        )

        if len(data["time"]) == 0:
            return data

    utils.write_netcdf_conventions_in_place(data)

    return data


def get_mndwi_threshold_no_cloud_images_near_date(
    site_name, geometry, max_cloud_cover, date_file,
    low_tide_search_days, mndwi_threshold: float = None,
    anomaly_filter_baseline: pandas.DataFrame = None,
    n_mad: float = 3.0,
):

    data = _get_mndwi_thresholded_images_near_date(
        site_name=site_name,
        geometry=geometry,
        date_file=date_file,
        low_tide_search_days=low_tide_search_days,
        mndwi_threshold=mndwi_threshold,
    )
        
    number_of_low_tide_dates = len(data["time"])

    if number_of_low_tide_dates == 0:
        return data

    data, cloud_cover_percentage, kept = filter_on_max_scl_ignore(data, max_cloud_cover, geometry)
    print(
        f"\tLow Cloud Tiles: {len(data['time'])} from the low tide tiles of "
        f"{number_of_low_tide_dates}. Cloud percentages {numpy.round(cloud_cover_percentage.data, decimals=1)} "
        f"Kept {kept.data}"
    )

    if len(data["time"]) == 0:
        return data

    if anomaly_filter_baseline is not None:
        data, deviation_per_band, kept = flag_anomalous_band_dates(
            data, baseline=anomaly_filter_baseline, n_mad=n_mad
        )

        if len(data["time"]) == 0:
            return data

    utils.write_netcdf_conventions_in_place(data)

    return data


def get_cloud_filtered_images_near_date(
    site_name, geometry, max_cloud_cover, date_file,
    low_tide_search_days,
    anomaly_filter_baseline: pandas.DataFrame = None,
    n_mad: float = 3.0,
):

    data = _get_images_near_date(
        site_name=site_name,
        geometry=geometry,
        date_file=date_file,
        low_tide_search_days=low_tide_search_days,
    )
        
    number_of_low_tide_dates = len(data["time"])

    if number_of_low_tide_dates == 0:
        return data

    data, cloud_cover_percentage, kept = filter_on_max_scl_ignore(data, max_cloud_cover, geometry)
    print(
        f"\tLow Cloud Tiles: {len(data['time'])} from the low tide tiles of "
        f"{number_of_low_tide_dates}. Cloud percentages {numpy.round(cloud_cover_percentage.data, decimals=1)} "
        f"Kept {kept.data}"
    )

    if len(data["time"]) == 0:
        return data

    if anomaly_filter_baseline is not None:
        data, deviation_per_band, kept = flag_anomalous_band_dates(
            data, baseline=anomaly_filter_baseline, n_mad=n_mad
        )

        if len(data["time"]) == 0:
            return data

    utils.write_netcdf_conventions_in_place(data)

    return data

def get_low_tide_no_cloud_images_in_year(
    geometry,
    max_cloud_cover,
    year: int, 
    low_tide_delta_hrs: int,
    low_tide_delta_mins: int, 
    anomaly_filter_baseline: pandas.DataFrame = None,
    n_mad: float = 3.0,
):

    data = _get_tide_api_filtered_images_in_year(
        geometry=geometry,
        year=year,
        low_tide_delta_hrs=low_tide_delta_hrs,
        low_tide_delta_mins=low_tide_delta_mins,
    )
    
    number_of_low_tide_dates = len(data["time"])

    data, cloud_cover_percentage, kept = filter_on_max_scl_ignore(data, max_cloud_cover, geometry)
    print(
        f"\tLow cloud tiles: {len(data['time'])} from the low tide tiles of"
        f" {number_of_low_tide_dates}. Cloud percentages {numpy.round(cloud_cover_percentage.data, decimals=1)} "
        f"Kept {kept.data}"
    )

    if len(data["time"]) == 0:
        return data

    if anomaly_filter_baseline is not None:
        data, deviation_per_band, kept = flag_anomalous_band_dates(
            data, baseline=anomaly_filter_baseline, n_mad=n_mad
        )

        if len(data["time"]) == 0:
            return data

    utils.write_netcdf_conventions_in_place(data)

    return data

def get_mndwi_and_cloud_filtered_images_in_year(
    geometry,
    max_cloud_cover,
    year: int, 
    mndwi_threshold: float = None,
    anomaly_filter_baseline: pandas.DataFrame = None,
    n_mad: float = 3.0,
):

    data = _get_mndwi_thresholded_images_in_year(
        geometry=geometry,
        year=year,
        mndwi_threshold=mndwi_threshold,
    )

    data, cloud_cover_percentage, kept = filter_on_max_scl_ignore(data, max_cloud_cover, geometry)
    print(
        f"\tLess than {max_cloud_cover}% cloud tiles: {len(data['time'])} from the MNDWI filtered tiles of"
        f" {len(kept)}. Cloud percentages {numpy.round(cloud_cover_percentage.data, decimals=1)} "
        f"Kept {kept.data}"
    )

    if len(data["time"]) == 0:
        return data

    if anomaly_filter_baseline is not None:
        data, deviation_per_band, kept = flag_anomalous_band_dates(
            data, baseline=anomaly_filter_baseline, n_mad=n_mad
        )

        if len(data["time"]) == 0:
            return data

    utils.write_netcdf_conventions_in_place(data)

    return data


def get_cloud_filtered_images_in_year(
    geometry,
    max_cloud_cover,
    year: int, 
    anomaly_filter_baseline: pandas.DataFrame = None, 
    n_mad: float = 3.0,
):

    print(f"\tGetting images")
    data = _get_images_in_year(
        geometry=geometry,
        year=year,
    )
    number_of_dates = len(data["time"])

    data, cloud_cover_percentage, kept = filter_on_max_scl_ignore(data, max_cloud_cover, geometry)
    print(
        f"\tLow cloud tiles: {len(data['time'])} from the total tiles of"
        f" {number_of_dates}. Cloud percentages {numpy.round(cloud_cover_percentage.data, decimals=1)} "
        f"Kept {kept.data}"
    )

    if len(data["time"]) == 0:
        return data

    if anomaly_filter_baseline is not None:
        data, deviation_per_band, kept = flag_anomalous_band_dates(
            data, baseline=anomaly_filter_baseline, n_mad=n_mad
        )

        if len(data["time"]) == 0:
            return data

    utils.write_netcdf_conventions_in_place(data)

    return data


def get_no_cloud_images_in_year(geometry, max_cloud_cover, year: int):
    """Return satellite images for the whole year with cloud cover at or below
    max_cloud_cover, with no low tide filtering applied (mirrors
    get_low_tide_no_cloud_images_in_year, but skips check_low_tide)."""

    geometry_WSG = geometry.buffer(S2_RESOLUTION).to_crs(
        utils.CRS_WSG
    )  # ensure includes pixles on edge

    search_collection = leafmap.stac_search(
        url=CATALOGUE_URL,
        max_items=400,
        collections=[COLLECTION],
        bbox=geometry_WSG.total_bounds,
        datetime=f"{year}-01-01/{year}-12-31",
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        get_collection=True,
    )

    items = search_collection.items
    # Optional solar-angle screen here: use item.properties["s2:mean_solar_zenith"] (elevation = 90 - zenith) and ["s2:mean_solar_azimuth"] before loading assets.

    data = odc.stac.load(
        items,
        bbox=geometry_WSG.total_bounds,
        bands=BANDS,
        chunks={},
        groupby="solar_day",
        resolution=S2_RESOLUTION,
        dtype="uint16",
        nodata=0,
        patch_url=planetary_computer.sign,
    )
    data = data.rio.clip(
        geometry.to_crs(data.rio.crs).geometry, all_touched=True, drop=True
    )
    data.load()
    data = harmonize_post_2022(data)

    print(f"\tTiles: {len(data['time'])} from a total of {len(items)} items")

    number_of_dates = len(data["time"])

    data, cloud_cover_percentage, kept = filter_on_max_scl_ignore(data, max_cloud_cover, geometry)
    print(
        f"\tNo cloud tiles: {len(data['time'])} from the tiles of"
        f" {number_of_dates}. Cloud percentages {numpy.round(cloud_cover_percentage.data, decimals=1)} "
        f"Kept {kept.data}"
    )

    if len(data["time"]) == 0:
        return data

    utils.write_netcdf_conventions_in_place(data)

    return data


def filter_valid_low_cloud_dates(data, max_cloud_cover):
    """Keep dates with site RGB coverage and acceptable cloud over those pixels."""
    valid = xarray.concat(
        [data[band].notnull() & (data[band] > 0) for band in RGB_BANDS], dim="rgb_band"
    ).all(dim="rgb_band")
    valid_count = valid.sum(dim=["x", "y"])
    cloud_count = (data["SCL"].isin(SCL_TO_IGNORE) & valid).sum(dim=["x", "y"])
    cloud_percentage = (100 * cloud_count / valid_count.where(valid_count > 0)).compute()
    kept = ((valid_count > 0) & (cloud_percentage <= max_cloud_cover)).compute()
    return data.isel(time=kept.values), cloud_percentage, kept



def filter_on_max_scl_ignore(data, max_cloud_cover, geometry):
    """Filter dates so that the percentage of ignored/bad pixels (SCL_TO_IGNORE
    or non-negative RGB) over the full site geometry does not exceed max_cloud_cover.
    """
    if len(data["time"]) == 0:
        return data, xarray.DataArray([]), xarray.DataArray([])

    # 1. Build polygon mask directly via rioxarray clip
    ones = xarray.ones_like(data["SCL"].isel(time=0), dtype="uint8")
    site_mask = (
        ones.rio.clip(geometry.to_crs(data.rio.crs).geometry, all_touched=True, drop=False)
        == 1
    )
    total_site_pixels = int(site_mask.sum())

    if total_site_pixels == 0:
        empty_kept = xarray.DataArray(numpy.zeros(data.sizes["time"], dtype=bool), dims="time")
        return data.isel(time=[]), xarray.DataArray([]), empty_kept

    # 2. Pixel is clean only if RGB is positive and SCL is not an ignored class
    rgb_valid = xarray.concat(
        [data[band].notnull() & (data[band] >= 0) for band in RGB_BANDS],
        dim="rgb_band",
    ).all(dim="rgb_band")
    scl_valid = data["SCL"].notnull() & ~data["SCL"].isin(SCL_TO_IGNORE)

    clean_pixels = (rgb_valid & scl_valid & site_mask).sum(dim=["x", "y"])

    # 3. Bad percentage calculated across entire site polygon
    bad_percentage = (
        100.0 * (total_site_pixels - clean_pixels) / total_site_pixels
    ).compute()
    kept = (bad_percentage <= max_cloud_cover).compute()

    return data.isel(time=kept.values), bad_percentage, kept

def compute_band_quality_baseline(
    training_sites: list,
    satellite_images_paths,
    bands: list = None,
    mad_shrinkage_k: float = 20.0,
) -> pandas.DataFrame:
    """Compute, per site and band, the median and MAD (median absolute deviation) of
    the per-date median reflectance, using only valid (non-nodata, non-ignored-SCL)
    pixels. Intended to be run once against the strictest available cloud subset
    (0% cloud) so the baseline itself isn't contaminated by the anomalous dates it
    will later be used to flag (see filter_valid_low_cloud_dates). Still requires low
    tide (not tide-agnostic), since tide state otherwise dominates the reflectance
    spread for intertidal sites. satellite_images_paths is a per-year satellite image
    folder (or list of folders, e.g. one per tide delta, to pool more dates per site
    , or for a mndwi thresholded images) from utils.get_prediction_path 
    (i.e. `<site>_<year>_sentinel-2.nc` files) - all years/folders found for a site
    are concatenated before computing the baseline, so the MAD reflects genuine
    date-to-date variation rather than a single date.

    Each site's raw MAD is shrunk toward a cross-site, n_images-weighted global MAD
    per band (shrunk = n/(n+k)*site_mad + k/(n+k)*global_mad, with k=mad_shrinkage_k),
    to stabilise the spread estimate for sites with few baseline images - sites with
    n_images >> k keep close to their own MAD, sites with few images lean on the
    shared baseline. Medians are left unshrunk since typical reflectance genuinely
    varies by site (substrate/turbidity). Set mad_shrinkage_k=0 to disable shrinkage.
    Returns a dataframe with columns site, band, median, mad, mad_raw, n_images -
    cache with utils.get_band_quality_baseline_path()."""

    if bands is None:
        bands = RGB_BANDS
    if isinstance(satellite_images_paths, (str, pathlib.Path)):
        satellite_images_paths = [satellite_images_paths]

    records = []
    for site_name in training_sites:
        satellite_files = []
        for satellite_images_path in satellite_images_paths:
            satellite_files.extend(sorted(satellite_images_path.glob(f"{site_name}_*_sentinel-2.nc")))
        if not satellite_files:
            print(
                f"\tWARNING - no baseline satellite images for site {site_name} under "
                f"{satellite_images_paths}. Skipping this site."
            )
            continue

        data = xarray.concat(
            [utils.load_satellite(filename=file, chunks=None) for file in satellite_files],
            dim="time",
        )
        valid = xarray.concat(
            [data[band].notnull() & (data[band] > 0) for band in RGB_BANDS], dim="rgb_band"
        ).all(dim="rgb_band") & ~data["SCL"].isin(SCL_TO_IGNORE)

        n_images = int(data.sizes["time"])
        if n_images == 0:
            print(f"\tWARNING - no dates available for site {site_name}. Skipping this site.")
            continue

        for band in bands:
            per_date_median = data[band].where(valid).median(dim=["x", "y"], skipna=True)
            site_median = float(per_date_median.median(dim="time", skipna=True))
            site_mad = float(
                numpy.abs(per_date_median - site_median).median(dim="time", skipna=True)
            )
            records.append({
                "site": site_name,
                "band": band,
                "median": site_median,
                "mad_raw": site_mad,
                "n_images": n_images,
            })

    baseline = pandas.DataFrame.from_records(records)
    if baseline.empty:
        return baseline

    if mad_shrinkage_k > 0:
        global_mad = baseline.groupby("band").apply(
            lambda group: numpy.average(group["mad_raw"], weights=group["n_images"])
        )
        shrinkage_weight = baseline["n_images"] / (baseline["n_images"] + mad_shrinkage_k)
        baseline["mad"] = (
            shrinkage_weight * baseline["mad_raw"]
            + (1 - shrinkage_weight) * baseline["band"].map(global_mad)
        )
    else:
        baseline["mad"] = baseline["mad_raw"]

    return baseline[["site", "band", "median", "mad", "mad_raw", "n_images"]]


def compute_and_save_band_quality_baseline(
    training_sites: list,
    satellite_images_paths,
    bands: list = None,
    mad_shrinkage_k: float = 20.0,
):
    """Compute the per-site, per-band quality baseline (see
    compute_band_quality_baseline) and write it to
    utils.get_band_quality_baseline_path() for reuse across inference runs."""

    baseline = compute_band_quality_baseline(
        training_sites=training_sites,
        satellite_images_paths=satellite_images_paths,
        bands=bands,
        mad_shrinkage_k=mad_shrinkage_k,
    )
    baseline_file = utils.get_band_quality_baseline_path()
    baseline.to_csv(baseline_file, index=False)
    print(f"\tSaved band quality baseline for {baseline['site'].nunique()} sites to {baseline_file}")
    return baseline


def load_band_quality_baseline() -> pandas.DataFrame:
    """Load the cached per-site, per-band quality baseline written by
    compute_and_save_band_quality_baseline."""
    return pandas.read_csv(utils.get_band_quality_baseline_path())


def _threshold_low_tide_dates(
    data,
    mndwi_threshold: float,
):
    """Drop dates whose mean MNDWI within the site polygon is greater than the threshold
    specified by the mndwi_threshold parameter. """
    mndwi = _mean_mndwi_by_date(data)
    valid = (mndwi <= mndwi_threshold)
    return data.isel(time=valid.values)

def _mean_mndwi_by_date(data):
    """Compute the mean MNDWI for each date within the site polygon. Exclude invalid
    pixels based on the SCL mask and negative reflectance values and zero denominators."""
    green = data["B03"].astype(numpy.float32)
    swir1 = data["B11"].astype(numpy.float32)
    valid_pixels = (
        data["SCL"].notnull()
        & ~data["SCL"].isin(SCL_TO_IGNORE)
        & (green >= 0)
        & (swir1 >= 0)
        & ((green + swir1) > 0)
    )

    mndwi = ((green - swir1) / (green + swir1)).where(valid_pixels)
    mean_mndwi = mndwi.mean(dim=["x", "y"], skipna=True)
    return mean_mndwi


def flag_anomalous_band_dates(
    data,
    baseline: pandas.DataFrame,
    n_mad: float = 3.0,
    bands: list = None,
):
    """Drop dates whose per-band median reflectance deviates from the cached
    per-site baseline (see compute_band_quality_baseline) by more than n_mad times
    the baseline MAD, e.g. to catch hazy/compressed dates that pass cloud filtering.
    Mirrors filter_valid_low_cloud_dates; returns (data_kept, deviation_per_band,
    kept), where deviation_per_band maps band -> DataArray of |median - baseline
    median| / baseline mad, and kept is True for dates within n_mad on every
    checked band."""

    if bands is None:
        bands = RGB_BANDS

    sites = baseline["site"].dropna().unique()
    if len(sites) != 1:
        raise ValueError("baseline must contain rows for exactly one site")
    site_name = sites[0]
    site_baseline = baseline[baseline["site"] == site_name].set_index("band")

    missing_bands = [band for band in bands if band not in site_baseline.index]
    if missing_bands:
        raise ValueError(
            f"No baseline for site {site_name} bands {missing_bands}. Run "
            "compute_and_save_band_quality_baseline first."
        )

    valid = xarray.concat(
        [data[band].notnull() & (data[band] > 0) for band in bands], dim="rgb_band"
    ).all(dim="rgb_band") & ~data["SCL"].isin(SCL_TO_IGNORE)

    deviation_per_band = {}
    kept = xarray.DataArray(numpy.ones(data.sizes["time"], dtype=bool), dims="time")
    for band in bands:
        band_median = float(site_baseline.loc[band, "median"])
        band_mad = float(site_baseline.loc[band, "mad"])
        if band_mad <= 0:
            print(f"\tWARNING - baseline MAD is 0 for site {site_name} band {band}. Skipping this band.")
            continue

        per_date_median = data[band].where(valid).median(dim=["x", "y"], skipna=True).compute()
        deviation = numpy.abs(per_date_median - band_median) / band_mad
        deviation_per_band[band] = deviation
        kept = kept & (deviation <= n_mad).fillna(False)

    kept = kept.compute()
    dates = data["time"].dt.strftime("%Y-%m-%d").values
    for time_index, date in enumerate(dates):
        band_deviations = ", ".join(
            f"{band}={float(deviation_per_band[band].isel(time=time_index)):.2f}"
            for band in deviation_per_band
        )
        status = "kept" if bool(kept.isel(time=time_index)) else "dropped"
        print(f"\t\t{date}: {band_deviations} -> {status}")

    print(
        f"\tAnomaly filter: {int(kept.sum())} of {data.sizes['time']} dates kept "
        f"for site {site_name} (n_mad={n_mad})"
    )

    return data.isel(time=kept.values), deviation_per_band, kept


def get_low_tide_info_from_tide_api(item, lat, lon):
    """Check if satellite images were taken during low tide."""

    dotenv.load_dotenv()
    tide_api_key = os.environ.get("TIDE_API", None)
    if tide_api_key is None:
        raise ValueError("TIDE_API environment variable not set in .env file")

    date_and_time = datetime.datetime.strptime(
        item.properties["datetime"], SATELLITE_DATE_FORMAT
    )
    start_date = (
        date_and_time - datetime.timedelta(hours=LOW_TIDE_DELTA)
        ).strftime(DATE_FORMAT_YYYYMMDD)
    tide_url = (
        f"{TIDE_API_STUB}?lat={lat}&long={lon}&datum=MSL"
        f"&numberOfDays=2&apikey={tide_api_key}&startDate={start_date}"
    )
    tide_query = requests.get(tide_url)
    tide_query.raise_for_status()
    tide_times = tide_query.json()["values"]
    time_from_low_tide = 12
    for tide_time in tide_times:
        if tide_time["value"] < 0:
            time_diff = abs(
                datetime.datetime.strptime(tide_time["time"], TIDE_DATE_FORMAT)
                - date_and_time
            )
            time_diff = int(time_diff / datetime.timedelta(hours=1))

            if time_diff < time_from_low_tide:
                time_from_low_tide = time_diff
    return time_from_low_tide


def _check_low_tide_from_tide_api(
    item, lat, lon, low_tide_delta_hrs: int, low_tide_delta_mins: int
):
    """Check if satellite images were taken during low tide. Lat and Lon should be in NZGD1949
    and the StartDate in the URL should be NZDT. """

    tide_info = get_low_tide_info_from_tide_api(item, lat, lon)

    return tide_info["time_from_low_tide"] <= pandas.Timedelta(
        hours=low_tide_delta_hrs, minutes=low_tide_delta_mins)


def get_low_tide_info_from_tide_api(item, lat, lon):
    """Returns the low tide information for the given satellite image and
      location - low tide height, time and time between the low tide and
      satellite image acquisition.
      lat: Latitude of the location in NZGD49
      lon: Longitude of the location in NZGD49
      """

    dotenv.load_dotenv()
    tide_api_key = os.environ.get("TIDE_API", None)
    if tide_api_key is None:
        raise ValueError("TIDE_API environment variable not set in .env file")

    satellite_datetime_utc = datetime.datetime.strptime(
        item.properties["datetime"],
        SATELLITE_DATE_FORMAT).replace(tzinfo=datetime.timezone.utc)
    satellite_datetime_nz_dt = satellite_datetime_utc.astimezone(
        zoneinfo.ZoneInfo('Pacific/Auckland'))
    
    start_date = ( # get the date 7hrs before the NZDT satellite time
        satellite_datetime_nz_dt - datetime.timedelta(hours=7)
        ).strftime(DATE_FORMAT_YYYYMMDD)
    tide_url = ( # Get tide for at least one low tide cyle either side
        f"{TIDE_API_STUB}?lat={lat}&long={lon}&datum=MSL"
        f"&numberOfDays=2&apikey={tide_api_key}&startDate={start_date}"
    )

    tide_query = requests.get(tide_url)
    tide_query.raise_for_status()
    
    tides = pandas.DataFrame.from_records(tide_query.json()["values"])
    tides["time"] = pandas.to_datetime(tides["time"], utc=True)
    tides = tides[tides["value"] < 0] # Low tides only

    time_from_low_tide = (tides["time"] - satellite_datetime_utc).abs().min()
    nearest_low = tides.loc[(tides["time"] - satellite_datetime_utc).abs().idxmin()]

    tide_info = { 
        "nearest_low_tide_height": nearest_low["value"],
        "nearest_low_tide_time": nearest_low["time"],
        "time_from_low_tide": time_from_low_tide 
            }

    return tide_info

def harmonize_post_2022(data):
    """Bring post-baseline reflectance bands onto the pre-2022 scale."""
    post_baseline = data["time"] > numpy.datetime64(HARMONIZE_DATE)
    for band in BANDS:
        if band == "SCL":
            continue
        reflectance = data[band]
        data[band] = xarray.where(
            post_baseline & reflectance.notnull() & (reflectance != 0),
            reflectance.clip(min=BAND_OFFSET_POST_2022_01_25) - BAND_OFFSET_POST_2022_01_25,
            reflectance,
        )
    return data


def get_satellite_info(geometry, date_YYMMDD):
    """Return item IDs and site-specific red, green, blue rescale lists for a date."""
    items = get_satellite_for_date(geometry=geometry, date_YYMMDD=date_YYMMDD)
    print(date_YYMMDD)
    tile_ids = []
    rescales_dict = {band: [] for band in RGB_BANDS}
    geometry_WSG = geometry.buffer(S2_RESOLUTION).to_crs(utils.CRS_WSG)
    for item in items:
        data = odc.stac.load(
            [item], bbox=geometry_WSG.total_bounds, bands=RGB_BANDS + ["SCL"],
            resolution=S2_RESOLUTION, dtype="uint16", nodata=0,
            patch_url=planetary_computer.sign,
        )
        try:
            data = data.rio.clip(
                geometry.to_crs(data.rio.crs).geometry, all_touched=True, drop=True
            )
        except NoDataInBounds:
            continue
        valid = ~data["SCL"].isin(SCL_TO_IGNORE)
        item_rescales = {}
        for band in RGB_BANDS:
            pixels = data[band].where(valid).values
            pixels = pixels[numpy.isfinite(pixels) & (pixels > 0)]
            if not pixels.size:
                break
            low, high = numpy.percentile(pixels, [2, 98])
            item_rescales[band] = (float(low), float(high))
        else:
            tile_ids.append(item.id)
            for band in RGB_BANDS:
                rescales_dict[band].append(item_rescales[band])
    if not tile_ids:
        raise ValueError(f"No satellite items with valid RGB pixels at this site on {date_YYMMDD}")
    return tile_ids, rescales_dict
            