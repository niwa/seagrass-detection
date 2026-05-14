"""Module focused on extracting training data and training pixel classification models."""

import utils
import sentinel2
import sampling
import gc
import geopandas
import xarray
import pandas
import scipy
import sklearn.ensemble
import joblib
import numpy
import pathlib
import matplotlib.pyplot


def train_classifier(
    training_sites: list,
    samples_path: pathlib.Path,
    uav_labels_file: pathlib.Path,
    uav_classes_to_ignore: dict,
    satellite_classes: dict,
    satellite_from_uav_classes: dict,
):
    """Combine all training datasets then drop the UAV classes to ignore
    and map the UAV classes to the satellite classes. Train a random
    forest classifier and return."""

    training_dataframe = map_satellite_ids_into_samples(
        training_sites=training_sites,
        samples_path=samples_path,
        uav_labels_file=uav_labels_file,
        uav_classes_to_ignore=uav_classes_to_ignore,
        satellite_classes=satellite_classes,
        satellite_from_uav_classes=satellite_from_uav_classes)

    print("\tTrain a Random Forest Model")
    training_classes = numpy.array(training_dataframe["satellite_class_id"])
    training_observations = numpy.array(
        training_dataframe.drop(columns=["satellite_class_id", "uav_class_id", "time"])
    )
    model_columns = pandas.DataFrame(columns=training_dataframe.drop(columns=["satellite_class_id", "uav_class_id", "time"]).columns)
    classifier = sklearn.ensemble.RandomForestClassifier()
    model = classifier.fit(training_observations, training_classes)

    return model, training_dataframe, model_columns


def load_samples(
    training_sites: list,
    samples_path: pathlib.Path,
) -> pandas.DataFrame:
    """Combine all training datasets and return."""

    print(f"\tLoad in sites: {training_sites}")
    samples_dataframe = []
    for training_site in training_sites:
        samples_file = samples_path / f"{training_site}_training_data.csv"
        samples_dataframe.append(pandas.read_csv(samples_file))
    samples_dataframe = pandas.concat(samples_dataframe, ignore_index=True)

    return samples_dataframe


def map_satellite_ids_into_samples(
    training_sites: list,
    samples_path: pathlib.Path,
    uav_labels_file: pathlib.Path,
    uav_classes_to_ignore: dict,
    satellite_classes: dict,
    satellite_from_uav_classes: dict,
):
    """Combine all training datasets then drop the UAV classes to ignore
    and map the UAV classes to the satellite classes."""

    samples_dataframe = load_samples(training_sites=training_sites,
                                      samples_path=samples_path)

    print("\tMap UAV training ids to the specified satellite training ids")
    uav_training_labels = (
        pandas.read_csv(uav_labels_file, sep="\t", header=None, names=["Value", "Key"])
        .set_index("Key")["Value"]
        .to_dict()
    )

    # Drop UAV image specific classes - e.g. Shadow, Glare
    class_ids_to_ignore = [uav_training_labels[key] for key in uav_classes_to_ignore]
    samples_dataframe = samples_dataframe[
        ~samples_dataframe["uav_class_id"].isin(class_ids_to_ignore)
    ]

    # Map the UAV classes to those used in the satellite imagery
    samples_dataframe["satellite_class_id"] = samples_dataframe["uav_class_id"]
    for key in satellite_from_uav_classes.keys():
        class_ids_to_map = [
            uav_training_labels[key] for key in satellite_from_uav_classes[key]
        ]
        samples_dataframe.loc[
            samples_dataframe["uav_class_id"].isin(class_ids_to_map),
            "satellite_class_id",
        ] = satellite_classes[key]
    return samples_dataframe


def predict_site(
    test_satellite_file: pathlib.Path,
    polygon_file: pathlib.Path,
    model_file: pathlib.Path,
    model_feature_names_file: pathlib.Path
):
    """Predict classes for satellite image across all time steps. Load feature names to ensure the same order"""

    satellite_data = utils.load_satellite(filename=test_satellite_file)
    uav_polygon = geopandas.read_file(polygon_file)
    model = joblib.load(model_file)
    model_columns = pandas.read_csv(model_feature_names_file)

    predictions = []
    print(f"\tPredict for {len(satellite_data['time'])} satellite images")
    for time_index in range(len(satellite_data["time"])):
        # Predictions
        observations_to_predict = (
            satellite_data.isel(time=time_index)[model_columns.columns]
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
    method_2_threshold,
):

    debug=False

    # Exit if the plots have already been created.
    overall_plot_filename = prediction_file.with_name(
            f"{prediction_file.stem}_confusion_matrix_time_all_dates.png"
        )
    if overall_plot_filename.exists():
        print(f"{overall_plot_filename.name} already exists."
              "Skipping. Delete plots if you want them regenerated.")
        return
    
    uav_training_labels = (
        pandas.read_csv(uav_labels_file, sep="\t", header=None, names=["Value", "Key"])
        .set_index("Key")["Value"]
        .to_dict()
    )

    print("Match satellite resolution to UAV then compare predictions to UAV")

    # Load in images
    uav_training_data = utils.load_classification(filename=test_uav_file, chunks=True, masked=False)
    sat_prediction_data = utils.load_satellite(filename=prediction_file)
    uav_polygon = geopandas.read_file(polygon_file)

    # drop classes to ignore
    uav_training_data_reclassed = uav_training_data.where(
        ~uav_training_data.isin(
            [uav_training_labels[key] for key in uav_classes_to_ignore]
        ),
        utils.UAV_NAN_CLASS,
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

    # Clip both to polygon
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
    all_truth = []
    all_predictions = []
    for time_index in range(len(sat_prediction_data["time"])):

        plot_filename = prediction_file.with_name(
            f"{prediction_file.stem}_confusion_matrix_time_{time_index}.png"
        )

        print(f"Time index: {time_index}")
        print(f"\tExtract truth and predictions")
        truth = uav_training_data_reclassed.values[
            sat_prediction_data.isel(time=time_index).notnull()
        ]
        predictions = sat_prediction_data.isel(time=time_index).values[
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
        all_truth.append(truth)
        all_predictions.append(predictions)

        if debug:
            if plot_filename.exists():
                print(f"{plot_filename.name} exists. Skipping. Delete if you want regenerated.")
                continue
            print(f"\tConstruct confusion matrix for time index")
            plot_confusion_matrix(truth=truth,
                                  predictions=predictions,
                                  class_names=satellite_classes,
                                  plot_filename=plot_filename,
                                  title= f"{int(method_2_threshold*100)}% Sampling Purity"; Time index {time_index}",
            )
            matplotlib.pyplot.close()

    # Force free memory
    del uav_training_data_reclassed
    del sat_prediction_data
    gc.collect()
    print("Overall confusion matrix across prediction dates")
    all_truth = numpy.concatenate(all_truth)
    all_predictions = numpy.concatenate(all_predictions)
    plot_confusion_matrix(truth=all_truth,
                          predictions=all_predictions,
                          class_names=satellite_classes,
                          plot_filename=overall_plot_filename,
                          title= f"{int(method_2_threshold*100)}% Sampling Purity",
    )

    return all_truth, all_predictions


def plot_model_feature_importance(training_dataframe, model_file):
    """Plot the feature importance of the trained random forest model."""

    plot_filename = model_file.with_name(f"{model_file.stem}_random_forest_feature_importance.png")
    if plot_filename.exists():
        print(f"{plot_filename.name} already exists. Delete if you've updated the "
              "model and want to regenerate")
    else:
        model = joblib.load(model_file)
        importance_df = pandas.DataFrame(
            {'Feature': training_dataframe.drop(columns=["satellite_class_id", "uav_class_id", "time"]).columns,
             'Importance': model.feature_importances_})
        importance_df.sort_values(by='Importance', ascending=False).plot(kind='bar', x='Feature', y='Importance')
        matplotlib.pyplot.savefig(model_file.with_name(f"{model_file.stem}_random_forest_feature_importance.png"), dpi=300)

def plot_uav_classes(training_dataframe, uav_labels_file):
    """Plot the UAV classes in the samples dataframe and return the figure"""
    uav_training_labels = (
        pandas.read_csv(uav_labels_file, sep="\t", header=None, names=["Value", "Key"])
        .set_index("Key")["Value"]
        .to_dict()
    )
    y_limits=(0, 6500)

    # Plot satellite bands for UAV classes
    number_uav_classes = len(training_dataframe["uav_class_id"].unique())
    nrows = int(numpy.ceil(number_uav_classes/3))
    figure, axes = matplotlib.pyplot.subplots(nrows=nrows, ncols=3, figsize=(21, 6*nrows))

    for i, (class_id, ax) in enumerate(zip(training_dataframe["uav_class_id"].unique(), axes.flat)):

        class_name = next((key for key, value in uav_training_labels.items() if value == class_id), None)

        training_dataframe[training_dataframe["uav_class_id"] == class_id].drop(columns=["SCL", "uav_class_id", "satellite_class_id"]).plot(kind='box', ax=ax, ylim=y_limits)
        ax.set_title(f"Spectral plot for class ID {class_name}")
    return figure

def plot_satellite_classes(training_dataframe, satellite_labels):
    """Plot the satellite classes in the samples dataframe and return the figure"""

    y_limits=(0, 6500)

    # Plot satellite bands for satellite classes
    nrows = int(numpy.ceil(len(satellite_labels)/3))
    figure, axes = matplotlib.pyplot.subplots(nrows=nrows, ncols=3, figsize=(21, 6*nrows))
    for i, (class_name, ax) in enumerate(zip(satellite_labels.keys(), axes.flat)):

        class_id = satellite_labels[class_name]

        training_dataframe[training_dataframe["satellite_class_id"] == class_id].drop(columns=["SCL", "satellite_class_id", "uav_class_id"]).plot(kind='box', ax=ax, ylim=y_limits)
        ax.set_title(f"Spectral plot for class ID {class_name}")
    return figure


def save_samples_uav_classes(plot_filename, training_dataframe, uav_labels_file):
    """Plot and save the UAV classes in the samples dataframe and return the figure"""
    figure = plot_uav_classes(training_dataframe=training_dataframe, uav_labels_file=uav_labels_file)
    figure.savefig(plot_filename, dpi=300)


def save_samples_satellite_classes(plot_filename, training_dataframe, satellite_labels):
    """Plot and save the satellite classes in the samples dataframe and return the figure"""
    figure = plot_satellite_classes(training_dataframe=training_dataframe, satellite_labels=satellite_labels)
    figure.savefig(plot_filename, dpi=300)


def plot_training_data_class_distribution(training_dataframe, model_file, uav_labels_file, satellite_labels):
    """Plot the class distribution of the training data."""

    # Plot satellite bands for UAV classes
    plot_filename = model_file.with_name(f"{model_file.stem}_training_uav_class_IDs.png")
    if plot_filename.exists():
        print(f"{plot_filename.name} already exists. Delete if you've updated the model"
              " and want to regenerate")
    else:
        save_samples_uav_classes(plot_filename=plot_filename,
                                 training_dataframe=training_dataframe,
                                 uav_labels_file=uav_labels_file)

    # Plot satellite bands for the satellite class used for prediction
    plot_filename = model_file.with_name(f"{model_file.stem}_training_satellite_class_IDs.png")
    if plot_filename.exists():
        print(f"{plot_filename.name} already exists. Delete if you've updated the model"
              " and want to regenerate")
    else:
        save_samples_satellite_classes(plot_filename=plot_filename,
                                 training_dataframe=training_dataframe,
                                 satellite_labels=satellite_labels)


def plot_confusion_matrix(
    truth,
    predictions,
    class_names: dict,
    plot_filename: pathlib.Path,
    title: str
):
    """ Create a confusion matrix with the class names """
    label_values = numpy.unique(numpy.concat([numpy.unique(truth), numpy.unique(predictions)]))
    label_names = [key for key, value in class_names.items() if value in label_values]

    confusion_matrix = sklearn.metrics.confusion_matrix(
        truth, predictions, normalize="true"
    )
    display = sklearn.metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=label_names
    )

    _, ax = matplotlib.pyplot.subplots(figsize=(10, 10))
    display.plot(
        ax=ax,
        cmap=matplotlib.pyplot.cm.Blues,
        values_format='.1%'
    )
    matplotlib.pyplot.xticks(rotation=90)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.savefig(plot_filename, dpi=300, )

# Not implemented yet
def confusion_matrix_of_site_satellite_resolution(
    test_uav_file,
    uav_labels_file,
    prediction_file,
    satellite_classes,
    satellite_from_uav_classes,
    uav_classes_to_ignore,
    polygon_file,
):

    debug=False
    # Exit if the plots have already been created.
    overall_plot_filename = prediction_file.with_name(
            f"{prediction_file.stem}_confusion_matrix_{sentinel2.S2_RESOLUTION}_resolution_time_all_dates.png"
        )
    if overall_plot_filename.exists():
        print(f"{overall_plot_filename.name} already exists."
              "Skipping. Delete plots if you want them regenerated.")
        return

    uav_training_labels = (
        pandas.read_csv(uav_labels_file, sep="\t", header=None, names=["Value", "Key"])
        .set_index("Key")["Value"]
        .to_dict()
    )

    print("Match UAV resolution to Satellite then compare predictions to UAV")

    # Load in images
    uav_training_data = utils.load_classification(filename=test_uav_file, chunks=True)
    sat_prediction_data = utils.load_satellite(filename=prediction_file)
    uav_polygon = geopandas.read_file(polygon_file)

    # drop classes to ignore
    uav_training_data_reclassed = uav_training_data.where(
        ~uav_training_data.isin(
            [uav_training_labels[key] for key in uav_classes_to_ignore]
        ),
        utils.UAV_NAN_CLASS,
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

    # Clip both to polygon
    uav_training_data_reclassed = uav_training_data_reclassed.rio.clip(
        uav_polygon.geometry, all_touched=True, drop=True
    )
    sat_prediction_data = sat_prediction_data.rio.clip(
        uav_polygon.geometry, all_touched=True, drop=True
    )

    #uav_training_data_reclassed.load()
    uav_training_data_reclassed, upsample_rate = sampling.align_fine_grid_to_coarse_grid(
        fine_grid=uav_training_data_reclassed, coarse_grid=sat_prediction_data)

        # Align UAV to satellite then coarsen taking the mode
    def get_mode(array, axis):
        """returns the mode values only; ignore counts"""
        return scipy.stats.mode(array, axis=axis).mode
    uav_training_data_reclassed = uav_training_data_reclassed.coarsen(
            x=upsample_rate, y=upsample_rate, boundary="trim"
            ).reduce(scipy.stats.mode)
    # Ensure exactly even spacing of the satellite imagry in x & y
    sat_prediction_data = sat_prediction_data.reindex_like(
        uav_training_data_reclassed, method="nearest"
    )

    # Pull out the predictions vs ground truth
    print("\tLoad UAV image")
    all_truth = []
    all_predictions = []
    for time_index in range(len(sat_prediction_data["time"])):

        plot_filename = prediction_file.with_name(
            f"{prediction_file.stem}_confusion_matrix_{sentinel2.S2_RESOLUTION}_resolution_time_{time_index}.png"
        )

        print(f"\tConstruct confusion matrix for time index: {time_index}")
        truth = uav_training_data_reclassed.values[
            sat_prediction_data.isel(time=time_index).notnull()
        ]
        predictions = sat_prediction_data.isel(time=time_index).values[
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

        all_truth.append(truth)
        all_predictions.append(predictions)

        if debug:
            if plot_filename.exists():
                print(f"{plot_filename.name} exists. Skipping. Delete if you want regenerated.")
                continue
            plot_confusion_matrix(truth=truth,
                                  predictions=predictions,
                                  class_names=satellite_classes,
                                  plot_filename=plot_filename,
                                  title=f"{int(method_2_threshold*100)}% Sampling Purity; Time index",
                                  )

    # Force free memory
    del uav_training_data_reclassed
    del sat_prediction_data
    gc.collect()

    print("\tOverall confusion matrix across prediction dates")
    all_truth = numpy.concatenate(all_truth)
    all_predictions = numpy.concatenate(all_predictions)
    plot_confusion_matrix(truth=all_truth,
                          predictions=all_predictions,
                          class_names=satellite_classes,
                          plot_filename=overall_plot_filename,
                          title= f"{int(method_2_threshold*100)}% Sampling Purity",
    )

    return all_truth, all_predictions
