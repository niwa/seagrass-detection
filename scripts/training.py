"""Module focused on extracting training data and training pixel classification models."""

import utils
import sentinel2
import gc
import geopandas
import xarray
import rioxarray
import pandas
import scipy
import sklearn.ensemble
import joblib
import numpy
import pathlib
import matplotlib.pyplot


def train_classifier(
    data_path: pathlib.Path,
    training_sites: list,
    training_path: pathlib.Path,
    uav_labels_file: pathlib.Path,
    uav_classes_to_ignore: dict,
    satellite_classes: dict,
    satellite_from_uav_classes: dict,
):
    """Combine all training datasets then drop the UAV classes to ignore
    and map the UAV classes to the satellite classes. Train a random
    forest classifier and return."""

    print(f"\tLoad in sites: {training_sites}")
    training_dataframe = []
    for training_site in training_sites:
        training_file = training_path / f"{training_site}_training_data.csv"
        training_dataframe.append(pandas.read_csv(training_file))
    training_dataframe = pandas.concat(training_dataframe, ignore_index=True)

    print("\tMap UAV training ids to the specified satellite training ids")
    uav_training_labels = (
        pandas.read_csv(uav_labels_file, sep="\t", header=None, names=["Value", "Key"])
        .set_index("Key")["Value"]
        .to_dict()
    )

    # Drop UAV image specific classes - e.g. Shadow, Glare
    class_ids_to_ignore = [uav_training_labels[key] for key in uav_classes_to_ignore]
    training_dataframe = training_dataframe[
        ~training_dataframe["uav_class_id"].isin(class_ids_to_ignore)
    ]

    # Map the UAV classes to those used in the satellite imagery
    training_dataframe["satellite_class_id"] = training_dataframe["uav_class_id"]
    for key in satellite_from_uav_classes.keys():
        class_ids_to_map = [
            uav_training_labels[key] for key in satellite_from_uav_classes[key]
        ]
        training_dataframe.loc[
            training_dataframe["uav_class_id"].isin(class_ids_to_map),
            "satellite_class_id",
        ] = satellite_classes[key]

    print("\tTrain a Random Forest Model")
    training_classes = numpy.array(training_dataframe["satellite_class_id"])
    training_observations = numpy.array(
        training_dataframe.drop(columns=["satellite_class_id", "uav_class_id", "time"])
    )
    classifier = sklearn.ensemble.RandomForestClassifier()
    model = classifier.fit(training_observations, training_classes)

    return model, training_dataframe


def predict_site(
    test_satellite_file: pathlib.Path,
    polygon_file: pathlib.Path,
    model_file: pathlib.Path,
):
    """Predict classes for satellite image across all time steps."""

    satellite_data = rioxarray.rioxarray.open_rasterio(
        test_satellite_file, parse_coordinates=True, masked=True
    )
    uav_polygon = geopandas.read_file(polygon_file)
    model = joblib.load(model_file)

    predictions = []
    print(f"\tPredict for {len(satellite_data['time'])} satellite images")
    for time_index in range(len(satellite_data["time"])):
        # Predictions
        observations_to_predict = (
            satellite_data.isel(time=time_index)
            .to_array()
            .stack(dims=["y", "x"])
            .transpose()
        )

        predictions_i = model.predict(observations_to_predict)
        # probabilities = model.predict_proba(observations_to_predict)
        predictions_i = predictions_i.reshape(
            len(satellite_data.y), len(satellite_data.x)
        )

        predictions_i = xarray.DataArray(
            [predictions_i],
            coords={
                "time": numpy.atleast_1d(satellite_data["time"][time_index]),
                "y": satellite_data.y,
                "x": satellite_data.x,
            },
            dims=["time", "y", "x"],
        )
        predictions.append(predictions_i)

    print("\tCombine predictions and clip to polygon")
    predictions = xarray.concat(predictions, dim="time")
    predictions.rio.write_crs(input_crs=utils.CRS_NZTM, inplace=True)
    predictions = predictions.rio.clip(
        uav_polygon.geometry, all_touched=True, drop=True
    )
    utils.write_netcdf_conventions_in_place(predictions)

    return predictions, satellite_data


def confusion_matrix_of_site(
    test_uav_file,
    uav_labels_file,
    prediction_file,
    satellite_classes,
    satellite_from_uav_classes,
    uav_classes_to_ignore,
    polygon_file,
):

    uav_training_labels = (
        pandas.read_csv(uav_labels_file, sep="\t", header=None, names=["Value", "Key"])
        .set_index("Key")["Value"]
        .to_dict()
    )

    print("\tLoad images and match resolution to UAV then compare predictions to UAV")

    # Load in images
    uav_training_data = rioxarray.rioxarray.open_rasterio(
        test_uav_file, parse_coordinates=True, masked=True, chunks=True
    ).squeeze("band", drop=True)
    sat_prediction_data = rioxarray.rioxarray.open_rasterio(
        prediction_file, parse_coordinates=True, masked=True
    )
    uav_polygon = geopandas.read_file(polygon_file)

    # drop classes to ignore
    uav_training_data_reclassed = uav_training_data.where(
        ~uav_training_data.isin(
            [uav_training_labels[key] for key in uav_classes_to_ignore]
        ),
        UAV_NAN_CLASS,
    )

    # convert UAV to satellite classifications
    for key in satellite_from_uav_classes.keys():
        class_ids_to_map = [
            uav_training_labels[key] for key in satellite_from_uav_classes[key]
        ]
        uav_training_data_reclassed = uav_training_data_reclassed.where(
            ~uav_training_data.isin(class_ids_to_map), satellite_classes[key]
        )

    # Force free memory
    del uav_training_data
    gc.collect()

    # Ensure nan for no data and clip both to polygon
    uav_training_data_reclassed.rio.set_nodata(numpy.nan)
    sat_prediction_data.rio.set_nodata(numpy.nan)

    uav_training_data_reclassed = uav_training_data_reclassed.rio.clip(
        uav_polygon.geometry, all_touched=True, drop=True
    )
    sat_prediction_data = sat_prediction_data.rio.clip(
        uav_polygon.geometry, all_touched=True, drop=True
    )

    # reindex to match training data
    sat_prediction_data = sat_prediction_data.reindex_like(
        uav_training_data_reclassed, method="nearest"
    )

    # Pull out the predictions vs ground truth
    print("\tLoad UAV image")
    all_truth = []
    all_predictions = []
    uav_training_data_reclassed.load()
    for time_index in range(len(sat_prediction_data["time"])):
        print(f"\tConstruct confusion matrix for time index: {time_index}")
        truth = uav_training_data_reclassed.data[
            sat_prediction_data.isel(time=0).notnull()
        ]
        predictions = sat_prediction_data.isel(time=time_index).data[
            sat_prediction_data.isel(time=time_index).notnull()
        ]
        # drop any NaN interpolated from clipped areas in the prediction
        mask = (
            ~numpy.isnan(predictions)
            & ~numpy.isnan(truth)
            & (truth != utils.UAV_NAN_CLASS)
        )
        truth = truth[mask]
        predictions = predictions[mask]

        confusion_matrix = sklearn.metrics.confusion_matrix(
            truth, predictions, normalize="true"
        )
        figure = sklearn.metrics.ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix, display_labels=satellite_classes.keys()
        )
        figure.plot(cmap=matplotlib.pyplot.cm.Blues)
        matplotlib.pyplot.savefig(
            prediction_file.with_name(
                f"{prediction_file.stem}_confusion_matrix_time_{time_index}.png"
            ),
            dpi=300,
        )
        matplotlib.pyplot.close()
        all_truth.append(truth)
        all_predictions.append(predictions)

    # Force free memory
    del uav_training_data_reclassed
    del sat_prediction_data
    gc.collect()

    print("\tOverall confusion matrix across prediction dates")
    all_truth = numpy.concatenate(all_truth)
    all_predictions = numpy.concatenate(all_predictions)
    confusion_matrix = sklearn.metrics.confusion_matrix(
        all_truth, all_predictions, normalize="true"
    )
    figure = sklearn.metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=satellite_classes.keys()
    )
    figure.plot(cmap=matplotlib.pyplot.cm.Blues)
    matplotlib.pyplot.savefig(
        prediction_file.with_name(
            f"{prediction_file.stem}_confusion_matrix_time_all_dates.png"
        ),
        dpi=300,
    )
