"""Unit tests for the pure/data-mapping helpers in scripts/training.py.

These avoid loading real satellite/UAV rasters or training models - they only
exercise functions that operate on plain numpy arrays, dataframes or small
xarray objects.
"""

import numpy
import pandas
import pytest
import torch
import xarray
import geopandas
from shapely.geometry import Polygon
from rasterio.transform import from_origin

import training


def test_unet_clip_marks_exterior_as_nodata_not_class_zero():
    predictions = xarray.DataArray(
        numpy.zeros((1, 4, 4), dtype=numpy.int8),
        dims=("time", "y", "x"),
        coords={
            "time": [numpy.datetime64("2025-01-01")],
            "y": [3.5, 2.5, 1.5, 0.5],
            "x": [0.5, 1.5, 2.5, 3.5],
        },
    )
    predictions.rio.write_crs("EPSG:2193", inplace=True)
    predictions.rio.write_transform(from_origin(0, 4, 1, 1), inplace=True)
    site_geometry = geopandas.GeoSeries(
        [Polygon([(0, 0), (4, 0), (0, 4), (0, 0)])], crs="EPSG:2193"
    )

    clipped = training._clip_unet_predictions_to_site(predictions, site_geometry)

    assert clipped.rio.nodata == training.utils.UAV_NAN_CLASS
    assert (clipped.values == training.utils.UAV_NAN_CLASS).any()
    assert (clipped.values == 0).any()


def test_extract_truth_and_predictions_excludes_nodata_from_both():
    truth = xarray.DataArray(
        [[0, training.utils.UAV_NAN_CLASS, 2, 3]],
        dims=("y", "x"),
    )
    predictions = xarray.DataArray(
        [[[0, 1, training.utils.UAV_NAN_CLASS, numpy.nan]]],
        dims=("time", "y", "x"),
    )

    valid_truth, valid_predictions = training.extract_truth_and_predictions(
        truth, predictions, time_index=0
    )

    numpy.testing.assert_array_equal(valid_truth, [0])
    numpy.testing.assert_array_equal(valid_predictions, [0])


def test_diffuser_unet_forward_uses_fixed_timestep_and_normalization(monkeypatch):
    calls = {}

    class FakeUNet(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            calls["configuration"] = kwargs

        def forward(self, tiles, timesteps):
            calls["tiles"] = tiles
            calls["timesteps"] = timesteps
            return type("Output", (), {"sample": tiles[:, :2]})()

    monkeypatch.setattr(training.diffusers, "UNet2DModel", FakeUNet)
    model = training.DiffuserUNetClassifier(
        in_channels=2, tile_size=64, num_classes=2,
        band_mean=[2, 3], band_std=[2, 1],
    )
    logits = model(torch.tensor([[[[float("nan")]], [[5.0]]]]))

    assert calls["configuration"]["out_channels"] == 2
    assert calls["timesteps"].tolist() == [0]
    torch.testing.assert_close(calls["tiles"], torch.tensor([[[[0.0]], [[2.0]]]]))
    assert logits.shape == (1, 2, 1, 1)


def test_diffuser_optimizer_uses_configured_learning_rate(monkeypatch):
    class FakeUNet(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(self, tiles, timesteps):
            return type("Output", (), {"sample": tiles})()

    monkeypatch.setattr(training.diffusers, "UNet2DModel", FakeUNet)
    model = training.DiffuserUNetClassifier(
        in_channels=2, tile_size=8, num_classes=2,
        band_mean=[0, 0], band_std=[1, 1], learning_rate=0.002,
    )

    optimizer = model.configure_optimizers()

    assert model.hparams.learning_rate == 0.002
    assert optimizer.param_groups[0]["lr"] == 0.002


@pytest.mark.parametrize("monitor_class_ids", [None, {"Seagrass": 1}])
def test_diffuser_validation_uses_only_selected_labels(monkeypatch, monitor_class_ids):
    logits = torch.tensor([[[[0., 0.], [0., 0.]],
                            [[4., 4.], [4., 0.]],
                            [[0., 0.], [0., 0.]]]])

    class FakeUNet(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, tiles, timesteps):
            return type("Output", (), {"sample": logits})()

    monkeypatch.setattr(training.diffusers, "UNet2DModel", FakeUNet)
    model = training.DiffuserUNetClassifier(
        in_channels=2, tile_size=2, num_classes=3,
        band_mean=[0, 0], band_std=[1, 1], monitor_class_ids=monitor_class_ids,
    )
    logged = {}
    monkeypatch.setattr(model, "log", lambda name, value, **kwargs: logged.setdefault(name, value))
    labels = torch.tensor([[[0, 1], [2, training.utils.UAV_NAN_CLASS]]])

    loss = model.validation_step((torch.zeros((1, 2, 2, 2)), labels), 0)

    expected_labels = labels.clone()
    if monitor_class_ids is not None:
        expected_labels[expected_labels != 1] = training.utils.UAV_NAN_CLASS
    torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(
        logits, expected_labels, ignore_index=int(training.utils.UAV_NAN_CLASS),
    ))
    expected_metrics = {"val_loss", "val_iou_Seagrass"} if monitor_class_ids else {
        "val_loss", "val_iou_0", "val_iou_1", "val_iou_2",
    }
    assert set(logged) == expected_metrics
    expected_iou = 1. if monitor_class_ids else 1. / 3.
    torch.testing.assert_close(
        logged["val_iou_Seagrass" if monitor_class_ids else "val_iou_1"],
        torch.tensor(expected_iou),
    )


def test_diffuser_validation_skips_batch_without_selected_labels(monkeypatch):
    class FakeUNet(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, tiles, timesteps):
            return type("Output", (), {"sample": torch.zeros(1, 2, 1, 1)})()

    monkeypatch.setattr(training.diffusers, "UNet2DModel", FakeUNet)
    model = training.DiffuserUNetClassifier(
        in_channels=2, tile_size=1, num_classes=2,
        band_mean=[0, 0], band_std=[1, 1], monitor_class_ids={"Seagrass": 1},
    )
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: pytest.fail("No metrics expected"))
    assert model.validation_step((torch.zeros(1, 2, 1, 1), torch.zeros(1, 1, 1, dtype=torch.long)), 0) is None


def test_diffuser_training_uses_separate_checkpoint_directory(monkeypatch, tmp_path):
    calls = {}

    def fake_train(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return "model", "checkpoint"

    monkeypatch.setattr(training, "_train_unet_classifier", fake_train)
    result = training.train_diffuser_unet_classifier(
        numpy.zeros((2, 6, 64, 64)), numpy.zeros((2, 64, 64)),
        tmp_path, num_classes=3, epochs=2,
    )

    assert result == ("model", "checkpoint")
    assert calls["args"][2] == tmp_path / "diffuser_unet"
    assert calls["kwargs"]["classifier_type"] is training.DiffuserUNetClassifier
    assert calls["kwargs"]["checkpoint_name"] == "diffuser-unet"
    assert calls["kwargs"]["epochs"] == 2


@pytest.mark.parametrize("hyper_parameters, expected_class", [
    ({"pretrained": True}, "UNetClassifier"),
    ({"tile_size": 64}, "DiffuserUNetClassifier"),
])
def test_load_unet_classifier_selects_architecture(
    monkeypatch, tmp_path, hyper_parameters, expected_class
):
    monkeypatch.setattr(training.torch, "load", lambda *args, **kwargs: {
        "hyper_parameters": hyper_parameters,
    })
    expected = object()
    classifier = getattr(training, expected_class)
    monkeypatch.setattr(
        classifier, "load_from_checkpoint", lambda path, **kwargs: expected
    )

    assert training.load_unet_classifier(
        tmp_path / "model.ckpt", weights_only=True
    ) is expected


@pytest.fixture
def uav_labels_file(tmp_path):
    """A small UAV class label lookup file: value<TAB>key per line."""
    labels_file = tmp_path / "uav_labels.txt"
    labels_file.write_text(
        "1\tSeagrass\n"
        "2\tSand\n"
        "3\tShadow\n"
        "4\tGlare\n"
        "5\tRock\n"
    )
    return labels_file


class TestMapSatelliteIdsIntoLabelsArray:
    def test_ignored_classes_become_nan_class(self, uav_labels_file):
        labels_array = numpy.array([1, 2, 3, 4, 5])

        result = training.map_satellite_ids_into_labels_array(
            labels_array=labels_array,
            uav_labels_file=uav_labels_file,
            uav_classes_to_ignore=["Shadow", "Glare"],
            satellite_classes={"Vegetation": 10, "Substrate": 20},
            satellite_from_uav_classes={
                "Vegetation": ["Seagrass"],
                "Substrate": ["Sand", "Rock"],
            },
        )

        numpy.testing.assert_array_equal(
            result, [10, 20, training.utils.UAV_NAN_CLASS, training.utils.UAV_NAN_CLASS, 20]
        )

    def test_result_is_int64(self, uav_labels_file):
        labels_array = numpy.array([1.0, 2.0])

        result = training.map_satellite_ids_into_labels_array(
            labels_array=labels_array,
            uav_labels_file=uav_labels_file,
            uav_classes_to_ignore=[],
            satellite_classes={"Vegetation": 10},
            satellite_from_uav_classes={"Vegetation": ["Seagrass"]},
        )

        assert result.dtype == numpy.int64

    def test_does_not_mutate_input_array(self, uav_labels_file):
        labels_array = numpy.array([1, 2])
        original = labels_array.copy()

        training.map_satellite_ids_into_labels_array(
            labels_array=labels_array,
            uav_labels_file=uav_labels_file,
            uav_classes_to_ignore=[],
            satellite_classes={"Vegetation": 10},
            satellite_from_uav_classes={"Vegetation": ["Seagrass"]},
        )

        numpy.testing.assert_array_equal(labels_array, original)

    def test_overlapping_output_and_input_ids_do_not_cause_double_mapping(
        self, uav_labels_file
    ):
        # Regression test: satellite class ids intentionally overlap with UAV
        # class ids ("Seagrass"=1 maps to 5, "Rock"=5 maps to 1). If the
        # implementation checked against the array being mutated in-place
        # instead of a preserved copy of the source labels, the second
        # mapping would incorrectly match pixels already rewritten by the
        # first mapping.
        labels_array = numpy.array([1, 5])

        result = training.map_satellite_ids_into_labels_array(
            labels_array=labels_array,
            uav_labels_file=uav_labels_file,
            uav_classes_to_ignore=[],
            satellite_classes={"ClassA": 5, "ClassB": 1},
            satellite_from_uav_classes={
                "ClassA": ["Seagrass"],
                "ClassB": ["Rock"],
            },
        )

        numpy.testing.assert_array_equal(result, [5, 1])

    def test_ignored_scl_classes_become_nan_when_enabled(self, uav_labels_file):
        labels_array = numpy.array([1, 2, 1, 2])
        scl_array = numpy.array([6, 8, 9, 10])

        result = training.map_satellite_ids_into_labels_array(
            labels_array=labels_array,
            uav_labels_file=uav_labels_file,
            uav_classes_to_ignore=[],
            satellite_classes={"Vegetation": 0, "Substrate": 1},
            satellite_from_uav_classes={
                "Vegetation": ["Seagrass"],
                "Substrate": ["Sand"],
            },
            scl_array=scl_array,
            drop_scl_classes_to_ignore=True,
        )

        numpy.testing.assert_array_equal(
            result,
            [0, training.utils.UAV_NAN_CLASS,
             training.utils.UAV_NAN_CLASS, training.utils.UAV_NAN_CLASS],
        )
        numpy.testing.assert_array_equal(labels_array, [1, 2, 1, 2])
        numpy.testing.assert_array_equal(scl_array, [6, 8, 9, 10])

    def test_scl_filter_is_opt_in_and_requires_matching_scl_array(self, uav_labels_file):
        labels_array = numpy.array([1, 2])
        mapping = {
            "labels_array": labels_array,
            "uav_labels_file": uav_labels_file,
            "uav_classes_to_ignore": [],
            "satellite_classes": {"Vegetation": 0, "Substrate": 1},
            "satellite_from_uav_classes": {
                "Vegetation": ["Seagrass"],
                "Substrate": ["Sand"],
            },
        }

        numpy.testing.assert_array_equal(
            training.map_satellite_ids_into_labels_array(**mapping), [0, 1]
        )
        with pytest.raises(ValueError, match="scl_array is required"):
            training.map_satellite_ids_into_labels_array(
                **mapping, drop_scl_classes_to_ignore=True
            )
        with pytest.raises(ValueError, match="same shape"):
            training.map_satellite_ids_into_labels_array(
                **mapping,
                scl_array=numpy.array([[6, 6]]),
                drop_scl_classes_to_ignore=True,
            )


class TestTimeIndexForDate:
    @pytest.fixture
    def dataset_with_dates(self):
        return xarray.Dataset(
            coords={
                "time": pandas.to_datetime(
                    ["2025-04-25", "2025-04-27", "2025-05-01"]
                )
            }
        )

    def test_finds_matching_date(self, dataset_with_dates):
        assert training.time_index_for_date(dataset_with_dates, "2025-04-27") == 1

    def test_ignores_time_of_day(self, dataset_with_dates):
        assert training.time_index_for_date(dataset_with_dates, "2025-04-25") == 0

    def test_raises_for_missing_date(self, dataset_with_dates):
        with pytest.raises(ValueError, match="No image for date"):
            training.time_index_for_date(dataset_with_dates, "2025-01-01")
