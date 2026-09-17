"""Module focused on plotting training data, model diagnostics and confusion matrices."""

import pandas
import sklearn.metrics
import joblib
import numpy
import pathlib
import re
import matplotlib.pyplot
import utils


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


def plot_validation_confusion_matrices(
    data_path: pathlib.Path,
    model_validation_label: str,
    method_2_threshold: float ,
):
    """Arrange validation confusion matrix PNGs in a parameter grid and save it."""
    validation_path = data_path / "validation"
    sampling_folder = utils.get_samples_folder(sample_method="sampling_2", method_2_threshold=method_2_threshold)

    folder_pattern = re.compile(
        r"low_tide_delta_(?:(?P<hours>\d+)hrs(?:_(?P<minutes>\d+)mins)?|"
        r"(?P<minutes_only>\d+)mins)_max_cloud_percentage_(?P<cloud_percentage>\d+)"
    )
    confusion_matrices = {}
    tide_delta_labels = {}

    for image_path in validation_path.glob(
        f"*/{sampling_folder}/{model_validation_label}*_confusion_matrix.png"
    ):
        folder_name = image_path.parent.parent.name
        folder_match = folder_pattern.fullmatch(folder_name)
        if folder_match is None:
            continue

        hours = int(folder_match.group("hours") or 0)
        if folder_match.group("minutes_only") is not None:
            minutes = int(folder_match.group("minutes_only"))
        else:
            minutes = int(folder_match.group("minutes") or 0)
        tide_delta_minutes = hours * 60 + minutes
        cloud_percentage = int(folder_match.group("cloud_percentage"))

        parameter_key = (tide_delta_minutes, cloud_percentage)
        if parameter_key in confusion_matrices:
            raise ValueError(
                f"Multiple confusion matrices found for {folder_name}"
            )
        confusion_matrices[parameter_key] = image_path
        tide_delta_labels[tide_delta_minutes] = " ".join(
            part for part in (f"{hours} hrs" if hours else "", f"{minutes} mins" if minutes else "")
            if part
        )

    if not confusion_matrices:
        raise FileNotFoundError(
            f"No {model_validation_label}*_confusion_matrix.png files found in "
            f"{validation_path}/*/{sampling_folder}"
        )

    tide_deltas = sorted(
        {tide_delta for tide_delta, _ in confusion_matrices}, reverse=True
    )
    cloud_percentages = sorted(
        {cloud_percentage for _, cloud_percentage in confusion_matrices}
    )
    figure, axes = matplotlib.pyplot.subplots(
        nrows=len(cloud_percentages),
        ncols=len(tide_deltas),
        figsize=(6 * len(tide_deltas), 6 * len(cloud_percentages)),
        squeeze=False,
    )

    for row_index, cloud_percentage in enumerate(cloud_percentages):
        for column_index, tide_delta in enumerate(tide_deltas):
            axis = axes[row_index, column_index]
            image_path = confusion_matrices.get((tide_delta, cloud_percentage))
            if image_path is not None:
                axis.imshow(matplotlib.pyplot.imread(image_path))
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row_index == 0:
                axis.set_title(f"Low tide delta: {tide_delta_labels[tide_delta]}")
            if column_index == 0:
                axis.set_ylabel(
                    f"Max cloud percentage: {cloud_percentage}%",
                    fontsize=12,
                )

    output_directory = validation_path / "overall_plots"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / (
        f"{sampling_folder}_{model_validation_label}_confusion_matrix.png"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    matplotlib.pyplot.close(figure)
    return output_path
