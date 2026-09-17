"""Module focused on downloading and processing Sentinel-2 data."""

import pathlib
import utils
import datetime
import requests
import odc.stac
import planetary_computer
import leafmap
import pandas
import numpy
import xarray
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
RGB_BANDS = ["B04", "B02", "B03"]
SCL_TO_IGNORE = [
    1,  # SATURATED_OR_DEFECTIVE
    8,  # CLOUD_MEDIUM_PROBABILITY
    9,  # CLOUD_HIGH_PROBABILITY
    10,  # THIN_CIRRUS
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


def get_low_tide_images_near_date(
    site_name, geometry, date_file, low_tide_search_days: int, low_tide_delta_hrs: int, low_tide_delta_mins: int
):
    """Return satellite images near the survey date near low tide."""
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
    lat = float(geometry_WSG.centroid.y.squeeze())
    lon = float(geometry_WSG.centroid.x.squeeze())
    all_tide_n = len(items)
    low_tide = []
    for item in items:
        low_tide.append(
            check_low_tide(item, lat=lat, lon=lon, low_tide_delta_hrs=low_tide_delta_hrs,
                           low_tide_delta_mins=low_tide_delta_mins)
        )

    items = [item for item, low_tide in zip(items, low_tide) if low_tide]

    if len(items) == 0:
        print(f"\tLow tide tiles: 0 from a total lowish cloud cover tiles of {all_tide_n}")
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
    print(
        f"\tLow tide tiles: {len(data['time'])} from a total lowish cloud "
        f"cover tiles of {all_tide_n}"
    )

    return data


def get_low_tide_images_in_year(geometry, year: int, low_tide_delta_hrs: int,
                                low_tide_delta_mins: int):
    """Reture satellite images near the survey date near low tide."""
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
    lat = float(geometry_WSG.centroid.y.squeeze())
    lon = float(geometry_WSG.centroid.x.squeeze())
    all_tide_n = len(items)
    low_tide = []
    for item in items:
        low_tide.append(
            check_low_tide(item, lat=lat, lon=lon, low_tide_delta_hrs=low_tide_delta_hrs,
                           low_tide_delta_mins=low_tide_delta_mins)
        )

    items = [item for item, low_tide in zip(items, low_tide) if low_tide]

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
    print(
        f"\tLow tide tiles: {len(data['time'])} from a total lowish cloud "
        f"cover tiles of {all_tide_n}"
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
    low_tide_search_days, low_tide_delta_hrs, low_tide_delta_mins,
):

    data = get_low_tide_images_near_date(
        site_name, geometry, date_file, low_tide_search_days,
        low_tide_delta_hrs, low_tide_delta_mins
    )
    number_of_low_tide_dates = len(data["time"])

    if number_of_low_tide_dates == 0:
        return data

    # keep only values with less cloud cover than the specified percentatge
    cloud_cover_percentage = (  # mean = % as mask is zeros and ones
        100
        * data["SCL"].isin(SCL_TO_IGNORE).mean(dim=["x", "y"]).compute()
    )
    data = data.where(
        cloud_cover_percentage <= max_cloud_cover,
        drop=True,
    )
    print(
        f"\tNo cloud tiles: {len(data['time'])} from the low tide tiles of "
        f"{number_of_low_tide_dates}. Cloud percentages {numpy.round(cloud_cover_percentage.data, decimals=1)} "
        f"Kept {cloud_cover_percentage.data <= max_cloud_cover}"
    )

    # Harmonize any post-2022 data - TODO - do for float32 too
    for index in range(len(data["time"])):
        date = pandas.Timestamp(
            data["time"].isel(time=index).values
        ).tz_localize("UTC").to_pydatetime()
        data_i = data.isel(time=index)
        harmonize_post_2022(data=data_i, date=date)
        utils.write_netcdf_conventions_in_place(data_i)

    return data


def get_low_tide_no_cloud_images_in_year(
    geometry, max_cloud_cover, year: int, low_tide_delta_hrs: int, low_tide_delta_mins: int
):

    data = get_low_tide_images_in_year(geometry, year, low_tide_delta_hrs=low_tide_delta_hrs,
                                       low_tide_delta_mins=low_tide_delta_mins)
    number_of_low_tide_dates = len(data["time"])

    # keep only values with less cloud cover than the specified percentatge
    cloud_cover_percentage = (
        100
        * data["SCL"].isin(SCL_TO_IGNORE).mean(dim=["x", "y"]).compute()
    )
    data = data.where(
        cloud_cover_percentage <= max_cloud_cover,
        drop=True,
    )
    print(
        f"\tNo cloud tiles: {len(data['time'])} from the low tide tiles of"
        f" {number_of_low_tide_dates}"
    )

    # Harmonize any post-2022 data - TODO - do for float32 too
    for index in range(len(data["time"])):
        date = pandas.Timestamp(
            data["time"].isel(time=index).values
        ).tz_localize("UTC").to_pydatetime()
        data_i = data.isel(time=index)
        harmonize_post_2022(data=data_i, date=date)
        utils.write_netcdf_conventions_in_place(data_i)

    return data


def get_low_tide(item, lat, lon):
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


def check_low_tide(item, lat, lon, low_tide_delta_hrs: int, low_tide_delta_mins: int = 0):
    """Check if satellite images were taken during low tide."""

    dotenv.load_dotenv()
    tide_api_key = os.environ.get("TIDE_API", None)
    if tide_api_key is None:
        raise ValueError("TIDE_API environment variable not set in .env file")

    date_and_time = datetime.datetime.strptime(
        item.properties["datetime"], SATELLITE_DATE_FORMAT
    )
    start_date = (
        date_and_time - datetime.timedelta(hours=low_tide_delta_hrs, minutes=low_tide_delta_mins)
        ).strftime(DATE_FORMAT_YYYYMMDD)
    tide_url = (
        f"{TIDE_API_STUB}?lat={lat}&long={lon}&datum=MSL"
        f"&numberOfDays=2&apikey={tide_api_key}&startDate={start_date}"
    )

    for attempt in range(1, 201):
        try:
            tide_query = requests.get(tide_url)
            tide_query.raise_for_status()
            break
        except requests.RequestException as error:
            if attempt == 200:
                raise RuntimeError(
                    f"Could not get the low tide API to respond after 200 attempts "
                    f"for item {item}."
                ) from error
            print(
                f"\tIgnore error {error} and try again in "
                f"{TIDE_API_RETRY_DELAY_SECONDS} second."
            )
            time.sleep(TIDE_API_RETRY_DELAY_SECONDS)

    tide_times = tide_query.json()["values"]
    low_tide = False
    for tide_time in tide_times:
        if tide_time["value"] < 0:
            time_diff = abs(
                datetime.datetime.strptime(tide_time["time"], TIDE_DATE_FORMAT)
                - date_and_time
            )
            if time_diff < datetime.timedelta(hours=low_tide_delta_hrs, minutes=low_tide_delta_mins):
                low_tide = True
                print(f"\tTime from low tide: {time_diff} for {item.id}")
                break
    return low_tide


def harmonize_post_2022(data, date, debug: bool = False):

    if numpy.datetime64(HARMONIZE_DATE) < numpy.datetime64(date):
        for band in BANDS:
            if debug:
                print(f"\t\tHarmonizing date {date}")
            data[band] = (
                data[band].clip(min=1000) - BAND_OFFSET_POST_2022_01_25
            )
    return data


def get_satellite_info(geometry, date_YYMMDD):
    items = get_satellite_for_date(geometry=geometry, date_YYMMDD=date_YYMMDD)

    tile_id = ""; percentage_2 = ""; percentage_98 = ""
    for item in items:
        tile_id += f"{item.id}, "
        stats=leafmap.stac_stats(collection=COLLECTION,
                                 item=item.id,
                                 titiler_endpoint="pc",
                                 assets=RGB_BANDS)
        percentage_2_i = []; percentage_98_i = []
        for key, value in stats.items():
            percentage_2_i.append(value['percentile_2'])
            percentage_98_i.append(value['percentile_98'])
        percentage_2_i = numpy.array(percentage_2_i).mean()
        percentage_98_i = numpy.array(percentage_98_i).mean()
        percentage_2 += f"{round(percentage_2_i)}, "
        percentage_98 += f"{round(percentage_98_i)}, "
    return tile_id, percentage_2, percentage_98
            