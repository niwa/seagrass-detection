"""Module containing utility functions for the project."""

import pathlib
import numpy
import shapely
import rasterio
import geopandas
import xarray
import rioxarray
import pandas

CRS_WSG = 4326
CRS_NZTM = 2193

UAV_NAN_CLASS = -128
CLASSIFICATION_DTYPE = numpy.int8

def get_data_path():
    """Get the path to the data folder."""
    return pathlib.Path(__file__).resolve().parent.parent / "data"


def get_samples_folder(sample_method: str, method_2_threshold: float):
    """Get the path to the sample folder for a given sampling method."""

    if sample_method == "sampling_1":
        sample_folder = sample_method
    elif sample_method == "sampling_2":
        sample_folder = f"{sample_method}_{int(100*method_2_threshold)}_percent"

    return sample_folder

def get_samples_path(sample_method: str, method_2_threshold: float, low_tide_delta: int, max_cloud_cover: int):
    """Get the path to the sample folder for a given sampling method."""

    sample_folder = get_samples_folder(sample_method, method_2_threshold)

    data_path = get_data_path()
    samples_path = data_path / "training" / f"low_tide_delta_{low_tide_delta}_max_cloud_percentage_{max_cloud_cover}" / sample_folder
    samples_path.mkdir(exist_ok=True, parents=True)
    return samples_path


def get_spectral_plots_path(sample_method: str, method_2_threshold: float, low_tide_delta: int, max_cloud_cover: int):
    """Get the path to the sample folder for a given sampling method."""

    sample_folder = get_samples_folder(sample_method, method_2_threshold)

    data_path = get_data_path()
    spectral_plots_path = data_path / "training" / f"low_tide_delta_{low_tide_delta}_max_cloud_percentage_{max_cloud_cover}" / "spectral_plots" / sample_folder
    spectral_plots_path.mkdir(exist_ok=True, parents=True)
    return spectral_plots_path


def get_samples_summary_file_path(sample_method: str, method_2_threshold: float, low_tide_delta: int, max_cloud_cover: int):
    """Get the path to the sample summary file for a given sampling method."""

    sample_folder_path = get_samples_path(sample_method, method_2_threshold, low_tide_delta, max_cloud_cover)
    return sample_folder_path / "samples_summary.csv"


def get_site_polygon_path(site_name: str):
    """Get the path to the site polygon file."""
    data_path = get_data_path()
    return data_path / "site_polygons" / f"{site_name}_polygon.gpkg"


def get_satellite_path(site_name: str, low_tide_delta: int, max_cloud_cover: int):
    """Get the path to the satellite file. Low_tide_delta in hrs, and max_cloud_cover
    as a percentage"""
    data_path = get_data_path()
    satellite_path = data_path / "satellite_images" / f"low_tide_delta_{low_tide_delta}_max_cloud_percentage_{max_cloud_cover}"
    satellite_path.mkdir(exist_ok=True)
    return satellite_path / f"{site_name}_sentinel-2.nc"


def get_training_data_path(site_name: str, sample_method: str, method_2_threshold: float, low_tide_delta: int, max_cloud_cover: int):
    """Get the path to the training data file."""
    sample_folder_path = get_samples_path(sample_method, method_2_threshold, low_tide_delta, max_cloud_cover)
    return sample_folder_path / f"{site_name}_training_data.csv"


def get_models_path(sample_method: str, method_2_threshold: float, low_tide_delta: int, max_cloud_cover: int):
    """Get the path to the sample folder for a given sampling method."""

    sample_folder = get_samples_folder(sample_method, method_2_threshold)

    data_path = get_data_path()
    models_path = data_path / "models" / f"low_tide_delta_{low_tide_delta}_max_cloud_percentage_{max_cloud_cover}" / sample_folder
    models_path.mkdir(exist_ok=True, parents=True)
    return models_path


def get_validation_path(sample_method: str, method_2_threshold: float, low_tide_delta: int, max_cloud_cover: int):
    """Get the path to the sample folder for a given sampling method."""

    sample_folder = get_samples_folder(sample_method, method_2_threshold)

    data_path = get_data_path()
    validation_path = data_path / "validation" /  f"low_tide_delta_{low_tide_delta}_max_cloud_percentage_{max_cloud_cover}" / sample_folder
    validation_path.mkdir(exist_ok=True, parents=True)
    return validation_path


def create_data_folders():
    """Create output folders for satellite, training,
    validation and predictions. No error if they already exist."""

    data_path = get_data_path()

    (data_path / "site_polygons").mkdir(exist_ok=True)

    (data_path / "satellite_images").mkdir(exist_ok=True, parents=True)

    (data_path / "training").mkdir(exist_ok=True, parents=True)

    (data_path / "models").mkdir(exist_ok=True,
                                                      parents=True)

    (data_path / "validation" / "predictions").mkdir(exist_ok=True,
                                                     parents=True)

    (data_path / "predictions" / "satellite_images").mkdir(exist_ok=True,
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

def read_uav_classe_labels_file(uav_class_labels_file: pathlib.Path):
    """Read in the UAV class lables file"""
    uav_class_labels = (
        pandas.read_csv(uav_class_labels_file, sep="\t", header=None, names=["Value", "Key"])
        .set_index("Key")["Value"]
        .to_dict()
    )
    return uav_class_labels


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


def ensure_grid_indexing(data: xarray.Dataset):
    """Order for correct display in GIS programs."""

    # Keep rows ordered north to south, as expected by raster GIS software.
    if data.y[0] < data.y[-1]:
        data = data.isel(y=slice(None, None, -1))
        write_netcdf_conventions_in_place(data)
        print("\tFlipped array to north-up orientation")
    return data


def save_netcdf(data: xarray.Dataset, filename):
    """Save rioxarray as a netcdf with compression
    and appropriate encoding."""

    data = ensure_grid_indexing(data)

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


def load_satellite(filename: pathlib, chunks: bool = None, masked: bool = True):
    """Load in a multiband satellite iamge."""
    satellite_data = rioxarray.rioxarray.open_rasterio(
            filename,
            parse_coordinates=True,
            masked=masked,
            chunks=chunks
        )
    return satellite_data


def load_classification(filename: pathlib, chunks: bool = True, masked: bool = True):
    """Load in a multiband satellite image."""
    classified_data = rioxarray.rioxarray.open_rasterio(
            filename,
            parse_coordinates=True,
            masked=masked,
            chunks=chunks, dtype="int8"
        )
    if "band" in classified_data.coords:
        classified_data = classified_data.squeeze("band", drop=True)
    return classified_data


def mask_to_polygons(mask_dataframe, coarsen_ratio: int = None):
    """
    Convert a rioxarray mask (DataArray) to a GeoDataFrame of polygons.
    Only True (or 1) values are converted to polygons. Note if coarsen
    specified apply to coarsen versioned of data frame
    """
    if coarsen_ratio is not None:
        mask_dataframe = mask_dataframe.coarsen(
            x=coarsen_ratio, y=coarsen_ratio, boundary="pad"
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
