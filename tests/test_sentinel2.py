"""Tests for Sentinel-2 site-specific display statistics."""

from types import SimpleNamespace

import numpy
import pandas
import pytest
import xarray

import sentinel2


def test_council_baseline_uses_training_reference_and_preserves_other_rows(tmp_path, monkeypatch):
    training_file = tmp_path / "training.csv"
    training = pandas.DataFrame([
        {"site": "Training1", "band": "B04", "mad_raw": 10.0, "n_images": 2},
        {"site": "Training2", "band": "B04", "mad_raw": 30.0, "n_images": 6},
    ])
    training.to_csv(training_file, index=False)
    original_training = training_file.read_bytes()
    council_file = tmp_path / "council_band_quality_baseline.csv"
    pandas.DataFrame([
        {"site": "Council", "band": "B04", "median": 1.0, "mad": 1.0, "mad_raw": 1.0, "n_images": 1},
        {"site": "Other", "band": "B04", "median": 2.0, "mad": 2.0, "mad_raw": 2.0, "n_images": 1},
    ]).to_csv(council_file, index=False)
    monkeypatch.setattr(sentinel2.utils, "get_data_path", lambda: tmp_path)
    monkeypatch.setattr(sentinel2, "compute_band_quality_baseline", lambda **kwargs: pandas.DataFrame([
        {"site": "Council", "band": "B04", "median": 100.0, "mad": 4.0, "mad_raw": 4.0, "n_images": 5},
    ]))

    result = sentinel2.compute_and_save_council_band_quality_baseline(
        council_sites=["Council"], satellite_images_paths=[], training_baseline_file=training_file,
    )

    assert training_file.read_bytes() == original_training
    assert len(result) == 2
    assert result.set_index("site").loc["Council", "mad"] == pytest.approx(20.8)
    assert result.set_index("site").loc["Other", "mad"] == 2.0
    assert len(result.set_index("site").loc["Council", "reference_sha256"]) == 64
    assert len(pandas.read_csv(council_file)) == 2


def test_council_baseline_does_not_overwrite_on_empty_result(tmp_path, monkeypatch):
    training_file = tmp_path / "training.csv"
    pandas.DataFrame([{"site": "Training", "band": "B04", "mad_raw": 10, "n_images": 2}]).to_csv(training_file, index=False)
    council_file = tmp_path / "council_band_quality_baseline.csv"
    original = b"site,band\nCouncil,B04\n"
    council_file.write_bytes(original)
    monkeypatch.setattr(sentinel2.utils, "get_data_path", lambda: tmp_path)
    monkeypatch.setattr(sentinel2, "compute_band_quality_baseline", lambda **kwargs: pandas.DataFrame())

    with pytest.raises(ValueError, match="No council"):
        sentinel2.compute_and_save_council_band_quality_baseline(["Council"], [], training_file)

    assert council_file.read_bytes() == original


def test_threshold_low_tide_dates_uses_only_site_polygon_pixels(monkeypatch):
    data = xarray.Dataset(
        {"SCL": (("time", "y", "x"), numpy.array([[[4, 6, 6], [6, 6, 6], [6, 6, 6]]]))},
        coords={
            "time": numpy.array(["2025-01-01"], dtype="datetime64[D]"),
            "y": [2.5, 1.5, 0.5],
            "x": [0.5, 1.5, 2.5],
        },
    ).rio.write_crs("EPSG:2193")

    class SitePolygon:
        geometry = [SimpleNamespace(__geo_interface__={})]

        def to_crs(self, crs):
            return self

    site_mask = numpy.array(
        [[True, True, False], [True, True, False], [False, False, False]]
    )
    monkeypatch.setattr(
        sentinel2.rasterio.features,
        "geometry_mask",
        lambda geometries, **kwargs: site_mask,
    )
    dry_thresholds = pandas.DataFrame(
        [{"site": "TestSite", "dry_threshold_percentage": 20}]
    )

    selected = sentinel2.threshold_low_tide_dates(
        data, "TestSite", dry_thresholds, SitePolygon()
    )

    assert selected.sizes["time"] == 0


def test_filter_valid_low_cloud_dates_excludes_empty_site_and_counts_cloud_over_coverage():
    rgb = numpy.array([[[0, 0, 0, 0]], [[100, 150, 200, 0]], [[100, 150, 200, 0]]])
    scl = numpy.array([[[0, 0, 0, 0]], [[6, 6, 9, 0]], [[6, 6, 9, 0]]])
    data = xarray.Dataset(
        {band: (("time", "y", "x"), rgb.copy()) for band in sentinel2.RGB_BANDS}
        | {"SCL": (("time", "y", "x"), scl)},
        coords={"time": numpy.array(["2016-04-06", "2016-05-06", "2016-06-06"], dtype="datetime64[D]")},
    )
    data["B02"][2, 0, :] = 0

    selected, cloud_percentage, kept = sentinel2.filter_valid_low_cloud_dates(data, 34)

    assert kept.values.tolist() == [False, True, False]
    assert selected.time.values.tolist() == data.time.values[1:2].tolist()
    assert numpy.isnan(cloud_percentage.values[0])
    assert cloud_percentage.values[1] == pytest.approx(100 / 3)


@pytest.mark.parametrize("download", ["near_date", "in_year"])
def test_download_discards_dates_without_rgb_pixels(monkeypatch, download):
    data = xarray.Dataset(
        {band: (("time", "y", "x"), numpy.zeros((1, 1, 1))) for band in sentinel2.BANDS},
        coords={"time": numpy.array(["2016-04-06"], dtype="datetime64[D]")},
    )
    source = "get_low_tide_images_near_date" if download == "near_date" else "get_low_tide_images_in_year"
    monkeypatch.setattr(sentinel2, source, lambda *args, **kwargs: data)
    monkeypatch.setattr(sentinel2.utils, "write_netcdf_conventions_in_place", lambda result: pytest.fail("Empty dataset must not be written"))

    if download == "near_date":
        result = sentinel2.get_low_tide_no_cloud_images_near_date("site", None, 0, None, 5, 0, 30)
    else:
        result = sentinel2.get_low_tide_no_cloud_images_in_year(None, 0, 2016, 0, 30)

    assert result.sizes["time"] == 0


def test_harmonize_post_2022_changes_returned_bands_but_not_scl():
    reflectance = numpy.array([
        [[450, 0, numpy.nan]],
        [[550, 0, numpy.nan]],
        [[1550, 0, numpy.nan]],
    ])
    data = xarray.Dataset(
        {band: (("time", "y", "x"), reflectance.copy()) for band in sentinel2.BANDS if band != "SCL"}
        | {"SCL": (("time", "y", "x"), numpy.full((3, 1, 3), 6))},
        coords={"time": numpy.array(["2022-01-24", "2022-01-25", "2022-02-01"], dtype="datetime64[D]")},
    )

    result = sentinel2.harmonize_post_2022(data)

    assert result is data
    for band in sentinel2.BANDS:
        if band == "SCL":
            numpy.testing.assert_array_equal(result[band].values, numpy.full((3, 1, 3), 6))
        else:
            numpy.testing.assert_allclose(
                result[band].values[:, 0, :],
                [[450, 0, numpy.nan], [550, 0, numpy.nan], [550, 0, numpy.nan]],
            )


@pytest.mark.parametrize("download", ["near_date", "in_year"])
def test_download_returns_harmonized_data(monkeypatch, download):
    data = xarray.Dataset(
        {band: (("time", "y", "x"), numpy.array([[[550]], [[1550]]]))
         for band in sentinel2.BANDS if band != "SCL"}
        | {"SCL": (("time", "y", "x"), numpy.array([[[6]], [[6]]]))},
        coords={"time": numpy.array(["2021-06-01", "2023-06-01"], dtype="datetime64[D]")},
    )
    source = "get_low_tide_images_near_date" if download == "near_date" else "get_low_tide_images_in_year"
    monkeypatch.setattr(sentinel2, source, lambda *args, **kwargs: data)
    monkeypatch.setattr(sentinel2.utils, "write_netcdf_conventions_in_place", lambda result: None)

    if download == "near_date":
        result = sentinel2.get_low_tide_no_cloud_images_near_date(
            "site", None, 100, None, 5, 1, 0,
        )
    else:
        result = sentinel2.get_low_tide_no_cloud_images_in_year(None, 100, 2023, 1, 0)

    assert result["B04"].values[:, 0, 0].tolist() == [550, 550]
    assert result["SCL"].values[:, 0, 0].tolist() == [6, 6]


@pytest.mark.parametrize("scl_values", ([6, 6, 9, 0], [0, 0, 9, 0]))
@pytest.mark.parametrize("item_ids", [
    ["empty", "outside", "first", "second"],
    ["empty", "outside"],
])
def test_get_satellite_info_uses_clear_rgb_pixels_within_site(monkeypatch, scl_values, item_ids):
    class Geometry:
        total_bounds = [0, 0, 1, 1]
        geometry = ["site"]

        def buffer(self, distance):
            return self

        def to_crs(self, crs):
            return self

    class Loaded:
        def __init__(self, site_data, outside=False):
            self.site_data = site_data
            self.outside = outside
            self.rio = self
            self.crs = "site-crs"

        def clip(self, geometry, all_touched, drop):
            if self.outside:
                raise sentinel2.NoDataInBounds("Site does not overlap item")
            return self.site_data

    def load(items, **kwargs):
        start = 100 if items[0].id == "first" else 300
        site_data = xarray.Dataset({
            band: (("time", "y", "x"), numpy.array([[[
                0 if items[0].id == "empty" else start + offset,
                0 if items[0].id == "empty" else start + offset + 100,
                0 if items[0].id == "empty" else 10000, 0,
            ]]]))
            for band, offset in zip(sentinel2.RGB_BANDS, (0, 20, 40))
        } | {"SCL": (("time", "y", "x"), numpy.array([[scl_values]]))})
        assert kwargs["bands"] == sentinel2.RGB_BANDS + ["SCL"]
        return Loaded(site_data, outside=items[0].id == "outside")

    monkeypatch.setattr(sentinel2, "get_satellite_for_date", lambda **kwargs: [
        SimpleNamespace(id=item_id) for item_id in item_ids
    ])
    monkeypatch.setattr(sentinel2.odc.stac, "load", load)

    if "first" not in item_ids:
        with pytest.raises(ValueError, match="No satellite items with valid RGB pixels"):
            sentinel2.get_satellite_info(Geometry(), "2025-01-01")
        return

    assert sentinel2.get_satellite_info(Geometry(), "2025-01-01") == (
        ["first", "second"],
        {
            "B04": [(102.0, 198.0), (302.0, 398.0)],
            "B03": [(122.0, 218.0), (322.0, 418.0)],
            "B02": [(142.0, 238.0), (342.0, 438.0)],
        },
    )