"""Tests for sparse table morphological operations."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import ndimage as ndi

from _skimage2.morphology._sparse_table import (
    FootprintDecomp,
    decomp_footprint,
    dilate,
    erode,
)


@pytest.fixture
def image():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (20, 20), dtype=np.uint8)


def _make_sparse_fp(shape):
    """Random 0/1 footprint with at least one nonzero cell."""
    rng = np.random.default_rng(sum(shape))
    fp = rng.integers(0, 2, shape, dtype=np.uint8)
    if not np.any(fp):
        fp[0, 0] = 1
    return fp


FP_SHAPES = [(3, 3), (5, 5), (4, 4), (4, 3), (3, 4)]
SCIPY_MODES = ["constant", "reflect", "mirror", "nearest", "wrap"]


class TestSparseTableVsScipy:
    """Sparse table erode/dilate must match scipy for all fp shapes and modes."""

    @pytest.mark.parametrize("fp_shape", FP_SHAPES)
    @pytest.mark.parametrize("fp_type", ["full", "sparse"])
    @pytest.mark.parametrize("mode", SCIPY_MODES)
    def test_erode_matches_scipy(self, image, fp_shape, fp_type, mode):
        fp = (
            np.ones(fp_shape, dtype=np.uint8)
            if fp_type == "full"
            else _make_sparse_fp(fp_shape)
        )
        decomp = decomp_footprint(fp)

        cval = int(np.iinfo(image.dtype).max)
        st_result = erode(image, decomp, mode=mode, cval=cval)
        sp_result = ndi.grey_erosion(image, footprint=fp, mode=mode, cval=cval)

        assert_array_equal(st_result, sp_result)

    @pytest.mark.parametrize("fp_shape", FP_SHAPES)
    @pytest.mark.parametrize("fp_type", ["full", "sparse"])
    @pytest.mark.parametrize("mode", SCIPY_MODES)
    def test_dilate_matches_scipy(self, image, fp_shape, fp_type, mode):
        fp = (
            np.ones(fp_shape, dtype=np.uint8)
            if fp_type == "full"
            else _make_sparse_fp(fp_shape)
        )
        decomp = decomp_footprint(fp)

        cval = int(np.iinfo(image.dtype).min)
        st_result = dilate(image, decomp, mode=mode, cval=cval)
        sp_result = ndi.grey_dilation(image, footprint=fp, mode=mode, cval=cval)

        assert_array_equal(st_result, sp_result)


class TestMathProperties:
    """Mathematical properties of erosion and dilation."""

    @pytest.mark.parametrize("fp_shape", [(3, 3), (5, 5)])
    def test_dilation_erosion_duality(self, image, fp_shape):
        """dilate(img, fp) == 255 - erode(255 - img, fp) for uint8.

        Only odd-size footprints are tested: for even-size fp, erode and
        dilate intentionally use different anchors (to match scipy), so the
        shift asymmetry breaks this duality.
        """
        fp = np.ones(fp_shape, dtype=np.uint8)
        decomp = decomp_footprint(fp)

        dilated = dilate(image, decomp)
        eroded_complement = erode(np.uint8(255) - image, decomp)

        assert_array_equal(dilated, np.uint8(255) - eroded_complement)


class TestDecompFootprint:
    """Sanity checks on FootprintDecomp structure."""

    def test_empty_footprint(self, image):
        """Empty footprint falls back to a default and does not raise."""
        fp = np.ones((0, 0), dtype=np.uint8)
        decomp = decomp_footprint(fp)
        result = erode(image, decomp)
        assert result.shape == image.shape

    def test_all_zero_footprint(self):
        """All-zero footprint: [0, 0] is forced to 1 internally."""
        fp = np.zeros((3, 3), dtype=np.uint8)
        decomp = decomp_footprint(fp)
        assert isinstance(decomp, FootprintDecomp)

    def test_dyadic_rects_structure(self):
        """dyadic_rects is a max_row_depth × max_col_depth nested list."""
        fp = np.ones((5, 7), dtype=np.uint8)
        decomp = decomp_footprint(fp)
        assert len(decomp.dyadic_rects) == decomp.plan_row.shape[0]
        for row in decomp.dyadic_rects:
            assert len(row) == decomp.plan_col.shape[1]
