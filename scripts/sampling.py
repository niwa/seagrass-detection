"""Methods to sample Sentinel-2 data where a uniform UAV classification exists."""


import utils
import sentinel2
import pathlib
import rioxarray
import xarray
import numpy
import pandas
import geopandas
import scipy.ndimage

def erosion_iterations_to_relate_rasters(fine_raster, coarse_raster):
    # coarse_area = res_coarse**2 = filter_width**2 * res_fine
    # filter_width = res_coarse / sqrt(res_fine)
    filter_width = int(
        max(numpy.abs(coarse_raster.rio.resolution()))
        / numpy.sqrt(max(numpy.abs(fine_raster.rio.resolution())))
    )
    # Ensure odd
    if filter_width % 2 == 0:  # Check if the number is even
        filter_width += 1
    # Calculate the number of iterations by a 3x3 filter
    number_of_iterations = int((filter_width - 3) / 2)
    return number_of_iterations


def extract_training_spectra_from_satellite_given_training_class(
    satellite: xarray.Dataset,
    training_raster: xarray.DataArray,
    training_labels: dict,
    class_key: str,
) -> pandas.DataFrame:
    """Use morpoholigical operations to try only take samples where there are
    only values matching the selected class. Doesn't seem to work quite right.
    """
    class_value = training_labels[class_key]

    # 1. Mask training raster to just selected class and resample to satellite
    # resolution with appropiate erosion
    number_of_iterations = erosion_iterations_to_relate_rasters(
        fine_raster=training_raster, coarse_raster=satellite
    )

    mask_fine = training_raster == training_labels[class_key]
    mask_fine.data = scipy.ndimage.binary_erosion(
        mask_fine,
        structure=numpy.ones((3, 3), dtype=bool),
        iterations=number_of_iterations,
    )
    mask_coarse = mask_fine.reindex_like(
        satellite[sentinel2.BANDS[0]], method="nearest"
    )

    if int(mask_coarse.sum()) == 0:
        return pandas.DataFrame({key: [] for key in satellite.data_vars})

    # 2. Remove unwanted SCL from class mask before sampling satellite.
    scl_mask = satellite["SCL"].isin(sentinel2.SCL_TO_IGNORE)
    mask_coarse = mask_coarse.where(~scl_mask, False)

    if int(mask_coarse.sum()) == 0:
        return pandas.DataFrame({key: [] for key in satellite.data_vars})

    # 3. Extract x and y only where to class values
    indicies_xy = numpy.argwhere(numpy.array(mask_coarse))
    x = numpy.array(mask_coarse.x[indicies_xy[:, 1]])
    y = numpy.array(mask_coarse.y[indicies_xy[:, 0]])
    xy_xarray = pandas.DataFrame({"x": x, "y": y}).to_xarray()

    # 4. Sample satellite - Get spectral info for training and add the class id
    training_spectrum = satellite.sel(xy_xarray).squeeze()
    if len(x) == 1:  # Ensure an index dimension so is not treated as a scaler
        training_spectrum = training_spectrum.expand_dims("index")
    training_spectrum = (
        training_spectrum.to_pandas()
        .drop(columns=["spatial_ref"])
        .assign(uav_class_id=class_value)
    )

    return training_spectrum


def training_data_from_images_method_2(satellite_data, uav_data, labels,
                                       threshold):
    """Extract training data from the images across the labels and
    return a pandas DataFrame"""

    # 1. Align UAV to satellite data but at finer resolution
    upsample_rate = round(
        satellite_data.rio.resolution()[0] / uav_data.rio.resolution()[0]
    )
    half_sat_res = satellite_data.rio.resolution()[0] / 2
    half_new_res = half_sat_res / upsample_rate
    edge_offset = half_sat_res - half_new_res
    resampled_y = numpy.linspace(
        satellite_data.y.data[0] + edge_offset,
        satellite_data.y.data[-1] - edge_offset,
        (len(satellite_data.y) - 0) * upsample_rate + 0,
        endpoint=True,
    )
    resampled_x = numpy.linspace(
        satellite_data.x.data[0] - edge_offset,
        satellite_data.x.data[-1] + edge_offset,
        (len(satellite_data.x) - 0) * upsample_rate + 0,
        endpoint=True,
    )
    uav_data = uav_data.reindex(x=resampled_x, y=resampled_y, method="nearest")

    # 3. Ensure exactly the same values for the satellite coordinates
    # - as observed e-10 differences in coordinate value spacing
    satellite_data = satellite_data.reindex_like(
        uav_data.coarsen(
            x=upsample_rate, y=upsample_rate, boundary="trim"
            ).count(),
        method="nearest",
    )

    training_spectrum = []
    for index in range(len(satellite_data["time"])):
        print(f"\tSample satellite {index + 1} of {len(satellite_data['time'])}")
        for class_key in labels.keys():

            # mask to label
            mask_fine = uav_data == labels[class_key]

            # coarsen mask and get mean - can then threshold 1 = all, .9 = 90%
            mask_coarse = mask_fine.coarsen(
                x=upsample_rate, y=upsample_rate, boundary="trim"
            ).mean()

            # mask from coarsened mask at thresold
            mask_coarse = mask_coarse >= threshold

            if int(mask_coarse.sum()) == 0:
                # Empty dataframe if no sites matchig the class
                sampled_spectrum = pandas.DataFrame(
                    {key: [] for key in satellite_data.isel(time=index).data_vars}
                )
            else:
                # Contruct xy array from indices where mask is True
                indicies_xy = numpy.argwhere(numpy.array(mask_coarse))
                x = numpy.array(mask_coarse.x[indicies_xy[:, 1]])
                y = numpy.array(mask_coarse.y[indicies_xy[:, 0]])
                xy_xarray = pandas.DataFrame({"x": x, "y": y}).to_xarray()

                # Sample the band values using the xy array
                sampled_spectrum = (
                    satellite_data.isel(time=index).sel(xy_xarray).squeeze()
                )
                if (
                    len(x) == 1
                ):  # Ensure an index dimension so is not treated as a scaler
                    sampled_spectrum = sampled_spectrum.expand_dims("index")
                sampled_spectrum = (
                    sampled_spectrum.to_pandas()
                    .drop(columns=["spatial_ref"])
                    .assign(uav_class_id=labels[class_key])
                )

            # Add to the other bands
            print(f"\t\tClass {class_key} - {len(sampled_spectrum)} samples")
            training_spectrum.append(sampled_spectrum)

    training_spectrum = pandas.concat(training_spectrum, ignore_index=True)
    return training_spectrum


def training_data_from_images_method_1(satellite_data, uav_data, labels):
    """Extract training data from the images across the labels and
    return a pandas DataFrame - doesn't require 100% to be of that class"""

    training_spectrum = []
    for index in range(len(satellite_data["time"])):
        print(f"\tSample satellite {index + 1} of {len(satellite_data['time'])}")
        for class_key in labels.keys():
            print(f"\t\tSample spectrum for {class_key}")
            sampled_spectrum = (
                extract_training_spectra_from_satellite_given_training_class(
                    satellite=satellite_data.isel(time=index),
                    training_raster=uav_data,
                    training_labels=labels,
                    class_key=class_key,
                )
            )
            print(f"\t{len(sampled_spectrum)} samples")
            training_spectrum.append(sampled_spectrum)

    training_spectrum = pandas.concat(training_spectrum, ignore_index=True)
    return training_spectrum


def sample_site(
    site_name: str,
    survey_dates_file: pathlib.Path,
    lowtide_search_range_file: pathlib.Path,
    training_labels_file: pathlib.Path,
    uav_folder: pathlib.Path,
    max_cloud_cover: float,
    sample_method: str,
    method_2_threshold: float = None,
):
    """Given a UAV file, get the corresponding satellite data and extract training data."""

    print(f"Site {site_name}")
    search_range = pandas.read_csv(lowtide_search_range_file)
    search_days = int(search_range["search_range"][search_range["site"]==site_name]) 
    training_labels = pandas.read_csv(
        training_labels_file, sep="\t", header=None, names=["Value", "Key"]
    ).set_index('Key')['Value'].to_dict()
    
    # load classified UAV - ensure is to nztm
    uav_file = uav_folder / f"{site_name}_classified.tif"
    if not uav_file.exists():
        print("\tWARNING - no classified image")
        raise ValueError(f"Missing classified image for site {site_name}")
    else:
        uav_data = rioxarray.rioxarray.open_rasterio(
            uav_file,
            parse_coordinates=True,
            masked=True,
            chunks=True
        ).squeeze("band", drop=True)
        
    # create or load area polygon
    polygon_file = utils.get_site_polygon_path(site_name)
    if not polygon_file.exists():
        print("\tsave polygon of the site")
        coarsen_ratio = max(numpy.ceil(sentinel2.S2_RESOLUTION / abs(numpy.array(uav_data.rio.resolution()))).astype(int))
        uav_polygon = utils.mask_to_polygons(uav_data != utils.UAV_NAN_CLASS, coarsen_ratio=coarsen_ratio)
        uav_polygon.to_file(polygon_file)
    else:
        uav_polygon = geopandas.read_file(polygon_file)
    
    # get or load low tide satellite with no cloud
    satellite_file = utils.get_satellite_training_path()
    if not satellite_file.exists():
        print("\tSave satellite image of the site around lowtide without cloud")
    
        satellite_data = sentinel2.get_low_tide_no_cloud_images_near_date(
            site_name=site_name,
            geometry=uav_polygon,
            max_cloud_cover=max_cloud_cover,
            date_file=survey_dates_file,
            low_tide_search_days=search_days
        )
        if len(satellite_data['time']) == 0:
            print("Warning: No satellite data without cloud. Ignore")
        else:
            utils.write_netcdf_conventions_in_place(satellite_data)
            satellite_data = satellite_data.rio.reproject(utils.CRS_NZTM)
            satellite_data = satellite_data.rio.clip(uav_polygon.geometry, drop=True, all_touched=True)
            utils.write_netcdf_conventions_in_place(satellite_data)
            utils.save_netcdf(satellite_data, satellite_file)
    else:
        satellite_data = rioxarray.rioxarray.open_rasterio(
            satellite_file,
            parse_coordinates=True,
            masked=True
        )

    # Extract training dataset
    training_file = utils.get_training_data_path(
        site_name=site_name, sample_method=sample_method, method_2_threshold=method_2_threshold
    )
    if not training_file.exists():
        print("Construct training data from UAV and Satellite imagery")

        if sample_method == "sampling_1":
            training_observations = training_data_from_images_method_1(
                satellite_data=satellite_data,
                uav_data=uav_data,
                labels=training_labels
            )
        elif sample_method == "sampling_2":
            training_observations = training_data_from_images_method_2(
                satellite_data=satellite_data,
                uav_data=uav_data,
                labels=training_labels, 
                threshold = method_2_threshold
            )
        else:
            raise ValueError(f"Invalid sample method: {sample_method}")
        training_observations.drop(columns=["x", "y"]).to_csv(training_file, index=False)
        # Save a gpkg of this
        training_observations_points = geopandas.GeoDataFrame(
            training_observations,
            geometry=geopandas.points_from_xy(training_observations.x, training_observations.y),
            crs=utils.CRS_NZTM
        )
        training_observations_points.to_file(training_file.with_suffix(".gpkg"))
        del training_observations_points
    else:
        training_observations = pandas.read_csv(training_file)
    
    summary=pandas.DataFrame(training_observations['uav_class_id'].value_counts())
    summary["uav_class_name"] = summary.index.map(lambda index: next((key for key, value in training_labels.items() if value == int(index)), None) )
    summary = summary[["uav_class_name", "count"]]
    summary.to_csv(training_file.with_stem(f"{training_file.stem}_summary"))
    summary

def site_sample_counts_by_class(sample_method: str, method_2_threshold: float):
    """Summarise the number of samples for each class for each site and save to csv."""
    sample_folder_path = utils.get_sample_folder_path(sample_method, method_2_threshold)
    counts_summary = []
    site_names = []
    for site_summary_file in sample_folder_path.glob("*_training_data_summary.csv"):
        site_names.append(site_summary_file.stem.replace("_training_data_summary", ""))
        counts_summary.append(pandas.read_csv(site_summary_file))
    # Into a summary across all sites
    counts_summary = pandas.concat(counts_summary, keys=site_names).reset_index(
        level=0, names="Site")[["Site","uav_class_name", "count"]].pivot(index="Site", columns="uav_class_name",values="count")
    counts_summary = counts_summary.fillna(0)
    counts_summary.to_csv(utils.get_samples_summary_file_path(sample_method, method_2_threshold), index=False)
    return counts_summary