"""Tests for site area timeseries model selection."""

import pytest
import plotly.graph_objects
import numpy
import pandas
import matplotlib.pyplot

import plotting
import utils


def test_classification_metrics_known_scores():
    per_class, overall = plotting.calculate_classification_metrics(
        [1, 1, 1, 1, 5, 5], [1, 1, 1, 5, 1, 5],
        {"Ulva": 5, "Seagrass": 1},
    )
    assert list(per_class.index) == ["Ulva", "Seagrass"]
    assert per_class.loc["Ulva", "support"] == 2
    assert per_class.loc["Seagrass", "support"] == 4
    for metric in ("precision", "recall", "f1"):
        assert per_class.loc["Ulva", metric] == pytest.approx(0.5)
        assert per_class.loc["Seagrass", metric] == pytest.approx(0.75)
    assert overall["accuracy"] == pytest.approx(4 / 6)
    assert overall["balanced_accuracy"] == pytest.approx(0.625)
    assert overall["macro_f1"] == pytest.approx(0.625)
    assert overall["valid_pixels"] == 6


def test_classification_metrics_masks_invalid_pairs_and_includes_absent_classes():
    per_class, overall = plotting.calculate_classification_metrics(
        [[1, numpy.nan, utils.UAV_NAN_CLASS], [1, 1, numpy.inf]],
        [[1, 5, 5], [utils.UAV_NAN_CLASS, numpy.inf, 1]],
        {"Seagrass": 1, "Ulva": 5},
    )
    assert per_class.loc["Ulva", "support"] == 0
    assert per_class.loc["Ulva", "f1"] == 0
    assert overall["accuracy"] == 1
    assert overall["balanced_accuracy"] == 1
    assert overall["macro_f1"] == 0.5
    assert overall["valid_pixels"] == 1


@pytest.mark.parametrize(
    ("truth", "predictions", "class_names", "message"),
    [
        ([1], [1, 1], {"Seagrass": 1}, "matching shapes"),
        ([], [], {"Seagrass": 1}, "No valid pixels"),
        ([utils.UAV_NAN_CLASS], [1], {"Seagrass": 1}, "No valid pixels"),
        ([1], [5], {"Seagrass": 1}, "present in class_names"),
        ([1], [1], {"Seagrass": 1, "Ulva": 1}, "unique"),
    ],
)
def test_classification_metrics_rejects_invalid_inputs(truth, predictions, class_names, message):
    with pytest.raises(ValueError, match=message):
        plotting.calculate_classification_metrics(truth, predictions, class_names)


def test_confusion_matrix_saves_metrics_with_matching_class_order(tmp_path, monkeypatch):
    plot_filename = tmp_path / "confusion_matrix.png"
    plot_filename.touch()
    assert not plotting.confusion_matrix_outputs_exist(plot_filename)
    displays = []
    display_type = plotting.sklearn.metrics.ConfusionMatrixDisplay

    def capture_display(**kwargs):
        displays.append(kwargs)
        return display_type(**kwargs)

    monkeypatch.setattr(plotting.sklearn.metrics, "ConfusionMatrixDisplay", capture_display)
    try:
        plotting.plot_confusion_matrix(
            [1, 1, 1, 1, 5, 5, numpy.nan], [1, 1, 1, 5, 1, 5, 1],
            {"Ulva": 5, "Seagrass": 1}, plot_filename, "Example",
        )
    finally:
        matplotlib.pyplot.close("all")
    assert displays[0]["display_labels"] == ["Ulva", "Seagrass"]
    numpy.testing.assert_allclose(
        displays[0]["confusion_matrix"], [[0.5, 0.5], [0.25, 0.75]]
    )
    assert plotting.confusion_matrix_outputs_exist(plot_filename)
    per_class = pandas.read_csv(tmp_path / "confusion_matrix_per_class_metrics.csv", index_col="class")
    overall = pandas.read_csv(tmp_path / "confusion_matrix_overall_metrics.csv", index_col=0)
    assert per_class.loc["Ulva", "f1"] == 0.5
    assert overall.loc["accuracy", "value"] == pytest.approx(4 / 6)


@pytest.mark.parametrize(
    ("options", "expected_models"),
    [
        ({}, {"RF", "UNet tile=128 stride=64", "UNet tile=64 stride=None"}),
        ({"include_unet": False}, {"RF"}),
        ({"include_rf": False}, {"UNet tile=128 stride=64", "UNet tile=64 stride=None"}),
    ],
)
def test_site_area_timeseries_selects_models(tmp_path, monkeypatch, options, expected_models):
    website_path = tmp_path / "website"
    model_names = (
        "RF_model_ihutai_trained_13_class_10_percent_test_train_sampling_2_80_percent_"
        "tide_30mins_cloud_1_predict_tide_30mins_cloud_0",
        "UN_model_all_trained_13_class_10_percent_test_100_epochs_128_tiling_64_stride_"
        "diffuser_unet_all_bands_all_SCL_train_sampling_2_80_percent_"
        "tide_30mins_cloud_1_predict_tide_30mins_cloud_0",
        "UN_model_all_trained_13_class_10_percent_test_25_epochs_64_tiling_32_stride_"
        "diffuser_unet_all_bands_all_SCL_train_sampling_2_80_percent_"
        "tide_1hrs_30mins_cloud_1_predict_tide_30mins_cloud_0",
        "UNet_model_256_64_10_percent_test_train_sampling_2_80_percent_"
        "tide_30mins_cloud_1_predict_tide_30mins_cloud_0",
    )
    for model_name in model_names:
        site_folder = website_path / model_name / "Example"
        site_folder.mkdir(parents=True)
        (site_folder / "info_all_dates.csv").write_text(
            "date,Seagrass Area [m^2]\n2024-01-01,10\n"
        )

    figures = {}
    monkeypatch.setattr(
        plotly.graph_objects.Figure,
        "write_html",
        lambda figure, path: figures.update({path.name: figure}),
    )

    output_paths = plotting.plot_site_predicted_vs_surveyed_areas(
        tmp_path, {}, targets=("Seagrass",), **options
    )

    assert len(output_paths) == 2
    site_figure = figures["Example_target_area_timeseries.html"]
    assert site_figure.layout.width == 2800
    assert {trace.name.split(", sampling=")[0] for trace in site_figure.data} == expected_models
    if options == {}:
        assert any("ihutai_trained_13_class" in trace.name for trace in site_figure.data)
        assert all(
            "all_trained_13_class_10_percent_test" in trace.name
            for trace in site_figure.data
            if trace.name.startswith("UNet")
        )
        assert any("model dt=90min" in trace.name for trace in site_figure.data)


def test_site_area_timeseries_ignores_legacy_folder_convention(tmp_path):
    website_path = tmp_path / "website"
    legacy_site = website_path / (
        "RF_model_10_percent_test_sampling_2_80_percent_low_tide_delta_30mins_"
        "max_cloud_percentage_1_predict_over_low_tide_delta_30mins_max_cloud_percentage_1"
    ) / "Example"
    legacy_site.mkdir(parents=True)
    (legacy_site / "info_all_dates.csv").write_text(
        "date,Seagrass Area [m^2]\n2024-01-01,10\n"
    )

    with pytest.raises(FileNotFoundError, match="No model prediction folders"):
        plotting.plot_site_predicted_vs_surveyed_areas(tmp_path, {}, targets=("Seagrass",))


def test_site_area_timeseries_requires_a_model(tmp_path):
    with pytest.raises(ValueError, match="At least one"):
        plotting.plot_site_predicted_vs_surveyed_areas(
            tmp_path, {}, include_rf=False, include_unet=False
        )