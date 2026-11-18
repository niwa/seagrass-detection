"""Unit tests for scripts/sampling.py grid helpers.

Uses small synthetic xarray DataArrays (with x/y coordinates recognised by
rioxarray) rather than real satellite/UAV rasters.
"""

import numpy
import pytest
import rioxarray  # noqa: F401 - registers the .rio accessor
import xarray

import sampling


def make_grid(resolution: float, size: int = 5, crs: int = 2193) -> xarray.DataArray:
    """Build a square DataArray on a regular x/y grid with the given pixel size."""
    x = numpy.arange(size) * resolution
    y = numpy.arange(size) * -resolution  # descending, as is conventional for rasters
    data = numpy.zeros((size, size), dtype="float32")
    grid = xarray.DataArray(data, coords={"y": y, "x": x}, dims=["y", "x"])
    grid.rio.write_crs(crs, inplace=True)
    return grid


class TestErosionIterationsToRelateRasters:
    def test_matching_resolution_needs_no_erosion(self):
        fine = make_grid(resolution=10)
        coarse = make_grid(resolution=10)

        iterations = sampling.erosion_iterations_to_relate_rasters(fine, coarse)

        assert iterations == 0

    def test_coarser_raster_requires_more_erosion_iterations(self):
        fine = make_grid(resolution=1)
        coarse = make_grid(resolution=10)

        iterations = sampling.erosion_iterations_to_relate_rasters(fine, coarse)

        # filter_width = 10 / sqrt(1) = 10 -> rounded up to odd (11)
        # iterations = (11 - 3) / 2 = 4
        assert iterations == 4


class TestAlignFineGridToCoarseGrid:
    def test_upsample_rate_matches_resolution_ratio(self):
        fine = make_grid(resolution=2, size=20)
        coarse = make_grid(resolution=10, size=5)

        aligned, upsample_rate = sampling.align_fine_grid_to_coarse_grid(
            fine_grid=fine, coarse_grid=coarse
        )

        assert upsample_rate == 5
        assert len(aligned.x) == len(coarse.x) * upsample_rate
        assert len(aligned.y) == len(coarse.y) * upsample_rate

    def test_aligned_grid_spans_similar_extent_to_coarse_grid(self):
        fine = make_grid(resolution=2, size=20)
        coarse = make_grid(resolution=10, size=5)

        aligned, _ = sampling.align_fine_grid_to_coarse_grid(
            fine_grid=fine, coarse_grid=coarse
        )

        assert aligned.x.min() >= coarse.x.min() - coarse.rio.resolution()[0]
        assert aligned.x.max() <= coarse.x.max() + coarse.rio.resolution()[0]
