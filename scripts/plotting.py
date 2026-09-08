"""Module focused on plotting training data, model diagnostics and confusion matrices."""

import pandas
import sklearn.metrics
import joblib
import numpy
import pathlib
import matplotlib.pyplot


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
