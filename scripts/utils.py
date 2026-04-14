"""Module containing utility functions for the project."""

import pathlib
import numpy
import shapely
import rasterio
import geopandas
import xarray

CRS_WSG = 4326
CRS_NZTM = 2193

UAV_NAN_CLASS = 0


def get_data_path():
    """Get the path to the data folder."""
    return pathlib.Path(__file__).resolve().parent / "data"


def get_sample_folder_path(sample_method: str, method_2_threshold: float):
    """Get the path to the sample folder for a given sampling method."""

    if sample_method == "sampling_1":
        sample_folder = sample_method
    elif sample_method == "sampling_2":
        sample_folder = f"{sample_method}_{int(100*method_2_threshold)}_percent"

    data_path = get_data_path()
    return data_path / "training" / "training_data" / sample_folder


def get_samples_summary_file_path(sample_method: str, method_2_threshold: float):
    """Get the path to the sample summary file for a given sampling method."""

    sample_folder_path = get_sample_folder_path(sample_method, method_2_threshold)
    return sample_folder_path / "samples_summary.csv"


def get_site_polygon_path(site_name: str):
    """Get the path to the site polygon file."""
    data_path = get_data_path()
    return data_path / "site_polygons" / f"{site_name}_polygon.gpkg"


def get_satellite_training_path(site_name: str):
    """Get the path to the satellite training file."""
    data_path = get_data_path()
    return data_path / "training" / "satellite_images" / f"{site_name}_sentinel-2.nc"


def get_training_data_path(site_name: str, sample_method: str, method_2_threshold: float):
    """Get the path to the training data file."""
    sample_folder_path = get_sample_folder_path(sample_method, method_2_threshold)
    return sample_folder_path / f"{site_name}_training_data.csv"


def get_prediction_path(prediction_name: str):
    """Get the path to the predicted data file."""
    data_path = get_data_path()
    return data_path / "validation" / "predictions" / prediction_name


def get_model_path(model_name: str):
    """Get the path to the model file."""
    data_path = get_data_path()
    return data_path / "models" / f"{model_name}.joblib"

def create_data_folders(data_path: pathlib.Path):
    """Create output folders for satellite, training,
    validation and predictions. No error if they already exist."""

    data_path = get_data_path()

    (data_path / "site_polygons").mkdir(exist_ok=True)

    (data_path / "training" / "satellite_images").mkdir(exist_ok=True,
                                                        parents=True)
    (data_path / "training" / "training_data").mkdir(exist_ok=True,
                                                     parents=True)
    (data_path / "training" / "training_data").mkdir(exist_ok=True,
                                                     parents=True)

    (data_path / "models" / "satellite_images").mkdir(exist_ok=True,
                                                      parents=True)

    (data_path / "validation" / "predictions").mkdir(exist_ok=True,
                                                     parents=True)

    (data_path / "predictions" / "satellite_images").mkdir(exist_ok=True,
                                                           parents=True)
    (data_path / "predictions" / "training_data").mkdir(exist_ok=True,
                                                        parents=True)
    (data_path / "predictions" / "predictions").mkdir(exist_ok=True,
                                                      parents=True)


def write_netcdf_conventions_in_place(data):
    """Write netcdf transform and crs in place for a xarray dataset
    or data array."""

    data.rio.transform(recalc=True)
    data.rio.write_transform(inplace=True)
    if isinstance(data, xarray.Dataset):
        for key in data.data_vars:
            data[key].rio.write_crs(data.rio.crs, inplace=True)
            """data[key].rio.write_nodata(
                data[key].rio.nodata, encoded=True, inplace=True
            )"""
    data.rio.write_crs(data.rio.crs, inplace=True)
    return data


def save_tiff(data: xarray.Dataset, filename):
    """Save rioxarray as a geotif with compression
    and appropriate encoding for geotiff conventions."""
    # print(f"\tsaving {filename.name}")
    data.encoding = {
        "dtype": data.dtype,
        "grid_mapping": data.encoding["grid_mapping"],
        "rasterio_dtype": data.dtype,
    }
    data.rio.to_raster(filename, compress="ZSTD", zstd_level=4)


def save_netcdf(data: xarray.Dataset, filename):
    """Save rioxarray as a netcdf with compression
    and appropriate encoding."""
    # print(f"\tsaving {filename.name}")
    encoding = {}
    if isinstance(data, xarray.Dataset):
        for key in data.data_vars:
            encoding[key] = {"zlib": True, "complevel": 9}
            if "grid_mapping" in data[key].encoding:
                encoding[key]["grid_mapping"] = (
                    data[key].encoding["grid_mapping"]
                )
    else:
        name = "__xarray_dataarray_variable__"
        encoding = {name: {"zlib": True, "complevel": 9}}
        if "grid_mapping" in data.encoding:
            encoding[name]["grid_mapping"] = data.encoding["grid_mapping"]
    data.to_netcdf(filename, format="NETCDF4",
                   engine="netcdf4", encoding=encoding)


def mask_to_polygons(mask_dataframe, coarsen_ratio: int = None):
    """
    Convert a rioxarray mask (DataArray) to a GeoDataFrame of polygons.
    Only True (or 1) values are converted to polygons. Note if coarsen
    specified apply to coarsen versioned of data frame
    """
    if coarsen_ratio is not None:
        mask_dataframe = mask_dataframe.coarsen(
            x=coarsen_ratio, y=coarsen_ratio, boundary="trim"
        ).max(skipna=True)
        mask_dataframe.rio.write_transform(
            mask_dataframe.rio.transform(recalc=True), inplace=True
        )

    mask = mask_dataframe.data.astype(numpy.uint8)
    transform = mask_dataframe.rio.transform()
    shapes = rasterio.features.shapes(mask, mask=mask.astype(bool),
                                      transform=transform)
    polygons = [
        shapely.geometry.shape(geom) for geom, value in shapes if value == 1
        ]
    polygon_dataframe = geopandas.GeoDataFrame(
        geometry=polygons, crs=mask_dataframe.rio.crs
    )
    return polygon_dataframe
