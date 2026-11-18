"""Unit tests for scripts/utils.py pure/path-formatting helpers."""

import numpy
import pytest

import utils


class TestFormatLowTideDelta:
    def test_hours_only(self):
        assert utils.format_low_tide_delta(1, 0) == "1hrs"

    def test_minutes_only(self):
        assert utils.format_low_tide_delta(0, 30) == "30mins"

    def test_hours_and_minutes(self):
        assert utils.format_low_tide_delta(1, 30) == "1hrs_30mins"

    def test_zero_returns_empty_string(self):
        assert utils.format_low_tide_delta(0, 0) == ""


class TestGetSamplesFolder:
    def test_sampling_1_ignores_threshold(self):
        assert utils.get_samples_folder("sampling_1", method_2_threshold=None) == "sampling_1"

    def test_sampling_2_includes_percentage(self):
        assert (
            utils.get_samples_folder("sampling_2", method_2_threshold=0.9)
            == "sampling_2_90_percent"
        )

    def test_sampling_2_rounds_down_percentage(self):
        # int() truncates rather than rounds - documenting current behaviour
        assert (
            utils.get_samples_folder("sampling_2", method_2_threshold=0.759)
            == "sampling_2_75_percent"
        )


class TestGetDataPaths:
    def test_get_data_path_points_at_repo_data_folder(self):
        data_path = utils.get_data_path()
        assert data_path.name == "data"
        assert data_path.exists()

    def test_get_training_satellite_path_builds_expected_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr(utils, "get_data_path", lambda: tmp_path)
        satellite_path = utils.get_training_satellite_path(
            site_name="MySite", low_tide_delta_hrs=1, low_tide_delta_mins=0, max_cloud_cover=5
        )
        assert satellite_path.name == "MySite_sentinel-2.nc"
        assert "tide_1hrs_cloud_5" in str(satellite_path)

    def test_thresholded_training_satellite_path_uses_threshold_mode_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr(utils, "get_data_path", lambda: tmp_path)
        satellite_path = utils.get_training_satellite_path(
            "MySite", 5, use_threshold=True,
        )

        assert satellite_path.parent.parent.name == "tide_thresholded_cloud_5"

    def test_thresholded_name_rejects_tide_deltas(self):
        with pytest.raises(ValueError, match="Tide deltas must be None"):
            utils.get_low_tide_max_cloud_name(
                5, low_tide_delta_hrs=1, use_threshold=True,
            )

    def test_get_samples_path_creates_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(utils, "get_data_path", lambda: tmp_path)
        samples_path = utils.get_samples_path(
            "sampling_2", 0.9, 5,
            low_tide_delta_hrs=1, low_tide_delta_mins=30,
        )
        assert samples_path.exists()
        assert samples_path.name == "sampling_2_90_percent"

    def test_filtered_training_paths_use_filtered_parent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(utils, "get_data_path", lambda: tmp_path)
        satellite_path = utils.get_training_satellite_path(
            site_name="MySite", low_tide_delta_hrs=1, low_tide_delta_mins=0,
            max_cloud_cover=5, filtered=True,
        )
        training_path = utils.get_training_data_path(
            site_name="MySite", sample_method="sampling_2", method_2_threshold=0.8,
            low_tide_delta_hrs=1, low_tide_delta_mins=0, max_cloud_cover=5,
            filtered=True,
        )

        assert satellite_path.parent.parent.name == "tide_1hrs_cloud_5_filtered"
        assert satellite_path.parent.name == "satellite_images"
        assert training_path.parent.parent.name == "tide_1hrs_cloud_5_filtered"

    def test_filtered_prediction_paths_use_filtered_parent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(utils, "get_data_path", lambda: tmp_path)
        prediction_path, satellite_path = utils.get_prediction_path(
            sample_method="sampling_2", method_2_threshold=0.8,
            model_low_tide_delta_hrs=1, model_low_tide_delta_mins=0,
            model_max_cloud_cover=5, predict_max_cloud_cover=10,
            predict_low_tide_delta_hrs=0, predict_low_tide_delta_mins=30,
            filtered=True,
        )

        assert satellite_path.parent.name == "tide_30mins_cloud_10_filtered"
        assert prediction_path.parent.name == "tide_30mins_cloud_10_filtered"
        assert prediction_path.name.endswith("_filtered")
        assert satellite_path.parent.parent == tmp_path / "predictions"

    @pytest.mark.parametrize("satellite_only", [False, True])
    def test_prediction_paths_use_council_root(self, tmp_path, monkeypatch, satellite_only):
        monkeypatch.setattr(utils, "get_data_path", lambda: tmp_path)
        prediction_root = tmp_path / "council_predictions"
        prediction_path, satellite_path = utils.get_prediction_path(
            sample_method="sampling_2", method_2_threshold=0.5,
            model_max_cloud_cover=5, predict_max_cloud_cover=1,
            predict_low_tide_delta_hrs=0, predict_low_tide_delta_mins=30,
            filtered=True, satellite_only=satellite_only,
            use_council_paths=True,
        )

        assert satellite_path == prediction_root / "tide_30mins_cloud_1_filtered" / "satellite_images"
        assert satellite_path.is_dir()
        if satellite_only:
            assert prediction_path is None
        else:
            assert prediction_path.parent == satellite_path.parent
            assert prediction_path.is_dir()

    def test_filtered_validation_path_uses_filtered_parent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(utils, "get_data_path", lambda: tmp_path)
        validation_path = utils.get_validation_path(
            sample_method="sampling_2", method_2_threshold=0.8,
            low_tide_delta_hrs=1, low_tide_delta_mins=0, max_cloud_cover=5,
            filtered=True,
        )

        assert validation_path.parent.name == "tide_1hrs_cloud_5_filtered"


class TestReadUavClassLabelsFile:
    def test_reads_tab_separated_value_key_pairs(self, tmp_path):
        labels_file = tmp_path / "labels.txt"
        labels_file.write_text("1\tSeagrass\n2\tSand\n")

        labels = utils.read_uav_classe_labels_file(labels_file)

        assert labels == {"Seagrass": 1, "Sand": 2}


class TestConstants:
    def test_uav_nan_class_is_negative(self):
        assert utils.UAV_NAN_CLASS == -128

    def test_classification_dtype_is_int8(self):
        assert utils.CLASSIFICATION_DTYPE == numpy.int8
