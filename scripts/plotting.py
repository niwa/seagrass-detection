"""Module focused on plotting training data, model diagnostics and confusion matrices."""

import pandas
import sklearn.metrics
import joblib
import numpy
import pathlib
import re
import matplotlib.pyplot
import plotly.colors
import plotly.graph_objects
import plotly.subplots
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


def _add_target_traces(
    figure, target, surveyed_dataframe, predictions, row, col, axis_number,
    survey_color, per_combination_labels, shown_legend_groups, predicted_color=None, predicted_colors=None
):
    """Add prediction + surveyed traces for one target to one subplot, holding the
    surveyed value flat out to both edges of that subplot's x-axis (no extra markers).

    Legend entries are de-duplicated via shown_legend_groups: when per_combination_labels
    is True, one entry per prediction label shared across all target subplots (since the
    label/survey line don't mention the target); otherwise one shared entry per target.
    """
    column = f"{target} Area [m^2]"
    for label, prediction_dataframe in predictions:
        if column not in prediction_dataframe.columns:
            continue
        group = label if per_combination_labels else f"{target}_predicted"
        color = predicted_colors.get(label) if predicted_colors is not None else predicted_color
        figure.add_trace(
            plotly.graph_objects.Scatter(
                x=prediction_dataframe["date"], y=prediction_dataframe[column],
                mode="lines+markers", line=dict(color=color),
                name=label if per_combination_labels else f"{target} (predicted)",
                legendgroup=group, showlegend=group not in shown_legend_groups,
            ),
            row=row, col=col
        )
        shown_legend_groups.add(group)

    if surveyed_dataframe is None or target not in surveyed_dataframe.columns:
        return

    surveyed_group = "surveyed" if per_combination_labels else f"{target}_surveyed"
    figure.add_trace(
        plotly.graph_objects.Scatter(
            x=surveyed_dataframe["date"], y=surveyed_dataframe[target],
            mode="lines+markers", line=dict(color=survey_color, dash="dot", shape="hv"),
            name="Surveyed (UAV)" if per_combination_labels else f"{target} (surveyed)",
            legendgroup=surveyed_group, showlegend=surveyed_group not in shown_legend_groups,
        ),
        row=row, col=col
    )
    shown_legend_groups.add(surveyed_group)

    dates = [surveyed_dataframe["date"]] + [
        prediction_dataframe["date"] for _, prediction_dataframe in predictions
    ]
    axis_min_date, axis_max_date = pandas.concat(dates).min(), pandas.concat(dates).max()
    first_date, first_value = surveyed_dataframe["date"].iloc[0], surveyed_dataframe[target].iloc[0]
    last_date, last_value = surveyed_dataframe["date"].iloc[-1], surveyed_dataframe[target].iloc[-1]
    for x0, x1, y in ((axis_min_date, first_date, first_value), (last_date, axis_max_date, last_value)):
        figure.add_shape(
            type="line", xref=f"x{axis_number}", yref=f"y{axis_number}",
            x0=x0, x1=x1, y0=y, y1=y,
            line=dict(color=survey_color, dash="dot"),
        )
    figure.update_xaxes(range=[axis_min_date, axis_max_date], row=row, col=col)


def plot_site_predicted_vs_surveyed_areas(
    data_path: pathlib.Path,
    targets: list = ("Seagrass", "Ulva", "Gracilaria"),
):
    """Plot, per site, predicted (across model/prediction parameter combinations) vs
    surveyed UAV target areas over time. Saves one interactive HTML per site, plus a
    composite summary HTML with one subplot per site."""
    website_path = data_path / "website"
    uav_areas_path = website_path / "uav_areas"

    folder_pattern = re.compile(
        r"RF_model_10_percent_test_sampling_2_(?P<method_2_threshold>\d+)_percent_"
        r"low_tide_delta_(?:(?P<model_hours>\d+)hrs(?:_(?P<model_minutes>\d+)mins)?|"
        r"(?P<model_minutes_only>\d+)mins)_max_cloud_percentage_(?P<model_cloud>\d+)_"
        r"predict_over_low_tide_delta_(?:(?P<predict_hours>\d+)hrs(?:_(?P<predict_minutes>\d+)mins)?|"
        r"(?P<predict_minutes_only>\d+)mins)_max_cloud_percentage_(?P<predict_cloud>\d+)"
    )

    model_folders = [
        folder for folder in website_path.iterdir()
        if folder.is_dir() and folder_pattern.fullmatch(folder.name) is not None
    ]
    if not model_folders:
        raise FileNotFoundError(f"No model prediction folders found in {website_path}")

    site_names = sorted({
        site_folder.name
        for model_folder in model_folders
        for site_folder in model_folder.iterdir()
        if site_folder.is_dir()
    })

    output_directory = website_path / "site_area_timeseries"
    output_directory.mkdir(parents=True, exist_ok=True)
    target_colors = {"Seagrass": "green", "Ulva": "orange", "Gracilaria": "purple"}
    target_survey_colors = {"Seagrass": "darkgreen", "Ulva": "chocolate", "Gracilaria": "indigo"}

    site_data = {}
    for site_name in site_names:
        surveyed_dataframe = None
        surveyed_csv = uav_areas_path / site_name / "uav_areas.csv"
        if surveyed_csv.exists():
            surveyed_dataframe = pandas.read_csv(surveyed_csv, parse_dates=["date"])

        predictions = []
        for model_folder in model_folders:
            info_csv = model_folder / site_name / "info_all_dates.csv"
            if not info_csv.exists():
                continue

            folder_match = folder_pattern.fullmatch(model_folder.name)
            model_hours = int(folder_match.group("model_hours") or 0)
            model_minutes = int(
                folder_match.group("model_minutes_only") or folder_match.group("model_minutes") or 0
            )
            predict_hours = int(folder_match.group("predict_hours") or 0)
            predict_minutes = int(
                folder_match.group("predict_minutes_only") or folder_match.group("predict_minutes") or 0
            )
            label = (
                f"model dt={model_hours * 60 + model_minutes}min "
                f"cloud<={folder_match.group('model_cloud')}%, "
                f"predict dt={predict_hours * 60 + predict_minutes}min "
                f"cloud<={folder_match.group('predict_cloud')}%"
            )
            predictions.append((label, pandas.read_csv(info_csv, parse_dates=["date"])))

        if surveyed_dataframe is None and not predictions:
            continue
        site_data[site_name] = (surveyed_dataframe, predictions)

    output_paths = []

    for site_name, (surveyed_dataframe, predictions) in site_data.items():
        palette = plotly.colors.qualitative.Plotly
        combination_colors = {label: palette[i % len(palette)] for i, (label, _) in enumerate(predictions)}
        figure = plotly.subplots.make_subplots(rows=len(targets), cols=1, subplot_titles=list(targets))
        shown_legend_groups = set()

        for row_index, target in enumerate(targets, start=1):
            _add_target_traces(
                figure, target, surveyed_dataframe, predictions,
                row=row_index, col=1, axis_number=row_index,
                survey_color="black", per_combination_labels=True,
                predicted_colors=combination_colors, shown_legend_groups=shown_legend_groups,
            )

        figure.update_layout(height=400 * len(targets), width=900, title=f"{site_name}: predicted vs surveyed areas")
        output_path = output_directory / f"{site_name}_target_area_timeseries.html"
        figure.write_html(output_path)
        output_paths.append(output_path)

    if site_data:
        ncols = 3
        nrows = int(numpy.ceil(len(site_data) / ncols))
        figure = plotly.subplots.make_subplots(
            rows=nrows, cols=ncols, subplot_titles=list(site_data.keys())
        )
        shown_legend_groups = set()

        for index, (site_name, (surveyed_dataframe, predictions)) in enumerate(site_data.items()):
            row, col = index // ncols + 1, index % ncols + 1
            for target in targets:
                _add_target_traces(
                    figure, target, surveyed_dataframe, predictions,
                    row=row, col=col, axis_number=index + 1,
                    predicted_color=target_colors.get(target), survey_color=target_survey_colors.get(target),
                    per_combination_labels=False, shown_legend_groups=shown_legend_groups,
                )

        figure.update_layout(height=400 * nrows, width=500 * ncols, title="Predicted vs surveyed target areas")
        composite_path = output_directory / "all_sites_target_area_timeseries.html"
        figure.write_html(composite_path)
        output_paths.append(composite_path)

    return output_paths
