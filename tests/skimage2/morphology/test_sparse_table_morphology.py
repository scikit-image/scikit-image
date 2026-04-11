"""Tests for sparse table morphological operations."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import ndimage as ndi

from _skimage2.morphology._sparse_table_morphology import (
    FootprintDecomp,
    _gen_dyadic_cover,
    _max_run_length_col,
    _max_run_length_row,
    _solve_rsap_greedy,
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


class TestPrivateHelpers:
    """Coverage of private helper edge cases unreachable via the public API."""

    def test_max_run_length_row_all_zeros(self):
        """All-zero footprint: no active runs → returns 0."""
        fp = np.zeros((3, 3), dtype=np.uint8)
        assert _max_run_length_row(fp) == 0

    def test_max_run_length_col_all_zeros(self):
        """All-zero footprint: no active runs → returns 0."""
        fp = np.zeros((3, 3), dtype=np.uint8)
        assert _max_run_length_col(fp) == 0

    def test_gen_dyadic_cover_oversized_depth(self):
        """Passing max_*_depth larger than the footprint needs exercises
        the early-break padding path (col and row directions)."""
        fp = np.ones((3, 3), dtype=np.uint8)
        result = _gen_dyadic_cover(fp, max_row_depth=4, max_col_depth=4)
        # Outer list has exactly max_row_depth entries, each of max_col_depth.
        assert len(result) == 4
        assert all(len(row) == 4 for row in result)

    def test_solve_rsap_greedy_col_path_for_max_i(self):
        """Pair (max_i, max_j) where max_i has higher col than the shared corner
        exercises the col-path body for max_i (line 312 of _sparse_table_morphology.py).

        initial_map entries at (r=0,c=0), (r=1,c=3), (r=3,c=1): the best
        pair is (r=1,c=3)&(r=3,c=1) with max_i at col=3 > max_x=1."""
        initial_map = np.zeros((4, 4), dtype=np.int8)
        initial_map[0, 0] = 1
        initial_map[1, 3] = 1  # (r=1, c=3)
        initial_map[3, 1] = 1  # (r=3, c=1)
        plan_row, plan_col, _ = _solve_rsap_greedy(initial_map)
        # The col path from (c=3,r=1) to shared corner (c=1,r=1) sets
        # plan_col[1, 2] and plan_col[1, 1] (two col steps).
        assert plan_col[1, 2]
        assert plan_col[1, 1]

    def test_solve_rsap_greedy_row_path_for_max_i(self):
        """After a swap-pop in round 1, pos order can invert so that max_i
        has a higher row than max_j, making the row-path body for max_i
        execute (line 314 of _sparse_table_morphology.py).

        initial_map with entries at the L-shaped corners causes exactly this."""
        initial_map = np.array(
            [[1, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.int8,
        )
        plan_row, plan_col, _ = _solve_rsap_greedy(initial_map)
        # Round 2 draws a row path from (c=0,r=2) down to (c=0,r=1),
        # setting plan_row[1, 0].
        assert plan_row[1, 0]


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

    def test_cross_5x5_matches_scipy(self):
        """5x5 cross exercises deeper plan trees (covers _solve_rsap_greedy
        path-drawing loops)."""
        fp = np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=np.uint8,
        )
        decomp = decomp_footprint(fp)
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, (20, 20), dtype=np.uint8)

        cval_e = int(np.iinfo(image.dtype).max)
        assert_array_equal(
            erode(image, decomp, mode="constant", cval=cval_e),
            ndi.grey_erosion(image, footprint=fp, mode="constant", cval=cval_e),
        )
        cval_d = int(np.iinfo(image.dtype).min)
        assert_array_equal(
            dilate(image, decomp, mode="constant", cval=cval_d),
            ndi.grey_dilation(image, footprint=fp, mode="constant", cval=cval_d),
        )

    def test_float_image(self):
        """Float32 image exercises the float branch of _neutral_cval."""
        rng = np.random.default_rng(1)
        image = rng.random((10, 10)).astype(np.float32)
        fp = np.ones((3, 3), dtype=np.uint8)
        decomp = decomp_footprint(fp)

        neutral = float(np.finfo(np.float32).max)
        result = erode(image, decomp, mode="constant", cval=neutral)
        expected = ndi.grey_erosion(image, footprint=fp, mode="constant", cval=neutral)
        np.testing.assert_array_almost_equal(result, expected)

    def test_col_heavier_plan_trim_ok(self):
        """A sparse footprint whose plan has a col-heavier root node exercises
        the else-branch in _morph_op's has_row-and-has_col block with
        _trim_ok=True (constant mode with neutral cval, lines 539-547, 552-553).

        Footprint chosen so that max_stack_depth[0,1] > max_stack_depth[1,0]."""
        fp = np.array(
            [[1, 0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 0, 1]],
            dtype=np.uint8,
        )
        decomp = decomp_footprint(fp)
        assert (
            decomp.max_stack_depth[0, 1] > decomp.max_stack_depth[1, 0]
        ), "footprint no longer produces a col-heavier plan; update test"
        rng = np.random.default_rng(2)
        image = rng.integers(0, 256, (20, 20), dtype=np.uint8)

        cval = int(np.iinfo(image.dtype).max)  # neutral for erosion → _trim_ok=True
        assert_array_equal(
            erode(image, decomp, mode="constant", cval=cval),
            ndi.grey_erosion(image, footprint=fp, mode="constant", cval=cval),
        )

    def test_col_heavier_plan_no_trim(self):
        """Same col-heavier footprint with a non-constant border mode
        exercises the _trim_ok=False sub-branch (lines 548-551)."""
        fp = np.array(
            [[1, 0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 0, 1]],
            dtype=np.uint8,
        )
        decomp = decomp_footprint(fp)
        rng = np.random.default_rng(3)
        image = rng.integers(0, 256, (20, 20), dtype=np.uint8)

        assert_array_equal(
            erode(image, decomp, mode="reflect"),
            ndi.grey_erosion(image, footprint=fp, mode="reflect"),
        )
