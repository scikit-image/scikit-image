import numpy as np
from numpy.testing import assert_equal
import pytest

from _skimage2._shared.testing import fetch, assert_stacklevel
from _skimage2.morphology import (
    footprints,
    cross_decompose_footprint,
    footprint_rectangle,
    footprint_ellipse,
    footprint_from_sequence,
    pad_footprint,
    mirror_footprint,
)


class TestFootprints:
    def strel_worker(self, fn, func):
        matlab_masks = np.load(fetch(fn))
        k = 0
        for arrname in sorted(matlab_masks):
            expected_mask = matlab_masks[arrname]
            actual_mask = func(k)
            if expected_mask.shape == (1,):
                expected_mask = expected_mask[:, np.newaxis]
            assert_equal(expected_mask, actual_mask)
            k = k + 1

    def strel_worker_3d(self, fn, func):
        matlab_masks = np.load(fetch(fn))
        k = 0
        for arrname in sorted(matlab_masks):
            expected_mask = matlab_masks[arrname]
            actual_mask = func(k)
            if expected_mask.shape == (1,):
                expected_mask = expected_mask[:, np.newaxis]
            # Test center slice for each dimension. This gives a good
            # indication of validity without the need for a 3D reference
            # mask.
            c = int(expected_mask.shape[0] / 2)
            assert_equal(expected_mask, actual_mask[c, :, :])
            assert_equal(expected_mask, actual_mask[:, c, :])
            assert_equal(expected_mask, actual_mask[:, :, c])
            k = k + 1

    def test_footprint_diamond(self):
        """Test diamond footprints"""
        self.strel_worker("data/diamond-matlab-output.npz", footprints.diamond)

    def test_footprint_ellipse_compare_matlab(self):
        """Compare behavior to Matlab."""
        file = "data/disk-matlab-output.npz"
        matlab_masks = np.load(fetch(file))
        for radius, name in enumerate(sorted(matlab_masks)):
            expected = matlab_masks[name]
            if expected.shape == (1,):
                expected = expected[:, np.newaxis]
            radii = (radius + .001,) * 2
            actual = footprint_ellipse(expected.shape, radii=radii)
            assert_equal(expected, actual)

    def test_footprint_octahedron(self):
        """Test octahedron footprints"""
        self.strel_worker_3d("data/diamond-matlab-output.npz", footprints.octahedron)

    def test_footprint_octagon(self):
        """Test octagon footprints"""
        expected_mask1 = np.array(
            [
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        actual_mask1 = footprints.octagon(5, 3)
        expected_mask2 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        actual_mask2 = footprints.octagon(1, 1)
        assert_equal(expected_mask1, actual_mask1)
        assert_equal(expected_mask2, actual_mask2)

    def test_footprint_ellipse_explicit_5_3(self):
        expected = np.array(
            [
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            ],
            dtype=np.uint8,
        )
        actual = footprint_ellipse((7, 11), radii=(4, 6), compare=np.less)
        assert_equal(expected, actual)

        # Switching dimensions makes no difference
        actual = footprint_ellipse((11, 7), radii=(6, 4), compare=np.less)
        assert_equal(expected, actual.T)

        # Large shape with same radii, can be croped to original result
        actual = footprint_ellipse((9, 15), radii=(4, 6), compare=np.less)
        actual = actual[1:-1, 2:-2]
        assert_equal(expected, actual)

    def test_footprint_ellipse_explicit(self):
        expected = np.ones((3, 3), dtype=np.uint8)
        actual = footprint_ellipse((3, 3))

        assert_equal(expected, actual)
        # assert_equal(expected, footprints.ellipse(3, 5).T)
        # assert_equal(expected, footprints.ellipse(1, 1).T)

    @pytest.mark.parametrize("shape", [(5, 7), (6, 6)])
    def test_footprint_ellipse_zero_radius(self, shape):
        footprint = footprint_ellipse(shape, radii=(2, 0))
        np.testing.assert_equal(footprint, 0)

        # For small non-zero radii the result depends on the "evenness" of
        # the dimension: if odd, the central column will always be 1, else 0
        expected = np.zeros(shape, dtype=np.uint8)
        if shape[1] % 2 == 1:
            expected[:, shape[1] // 2] = 1  # Central axis of 1 only in odd case
        footprint = footprint_ellipse(shape, radii=(2, np.nextafter(0, 1)))
        np.testing.assert_equal(footprint, expected)

    def test_footprint_star(self):
        """Test star footprints"""
        expected_mask1 = np.array(
            [
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        actual_mask1 = footprints.star(4)
        expected_mask2 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
        actual_mask2 = footprints.star(1)
        assert_equal(expected_mask1, actual_mask1)
        assert_equal(expected_mask2, actual_mask2)


@pytest.mark.parametrize(
    'function, args, supports_sequence_decomposition',
    [
        (footprints.diamond, (3,), True),
        (footprints.octahedron, (3,), True),
        (footprint_rectangle, ((3, 5),), True),
        (footprints.octagon, (3, 4), True),
        (footprints.star, (3,), False),
    ],
)
@pytest.mark.parametrize("dtype", [np.uint8, np.float64])
def test_footprint_dtype(function, args, supports_sequence_decomposition, dtype):
    # make sure footprint dtype matches what was requested
    footprint = function(*args, dtype=dtype)
    assert footprint.dtype == dtype

    if supports_sequence_decomposition:
        sequence = function(*args, dtype=dtype, decomposition='sequence')
        assert all([fp_tuple[0].dtype == dtype for fp_tuple in sequence])


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("radius", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20])
def test_nsphere_series_approximation(ndim, radius):
    shape = (radius * 2 + 1,) * ndim
    expected = footprint_ellipse(shape)
    decomposed = footprint_decomposed_disk(radius, ndim=ndim)
    approximate = footprints.footprint_from_sequence(decomposed)
    assert approximate.shape == expected.shape

    # verify that maximum error does not exceed some fraction of the size
    error = np.sum(np.abs(expected.astype(int) - approximate.astype(int)))
    if radius == 1:
        assert error == 0
    else:
        max_error = 0.1 if ndim == 2 else 0.15
        assert error / expected.size <= max_error


@pytest.mark.parametrize("radius", [1, 2, 3, 4, 5, 10, 20, 50, 75])
@pytest.mark.parametrize("dtype", [bool, np.uint8, int, float])
def test_ellipse_crosses_approximation_(radius, dtype):
    shape = (radius * 2 + 1,) * 2
    expected = footprint_ellipse(shape, dtype=dtype)
    decomposed = cross_decompose_footprint(expected)
    approximate = footprint_from_sequence(decomposed)
    assert approximate.shape == expected.shape
    assert approximate.dtype == dtype

    # verify that maximum error does not exceed some fraction of the size
    error = np.sum(np.abs(expected.astype(int) - approximate.astype(int)))
    max_error = 0.01
    assert error / expected.size <= max_error


@pytest.mark.parametrize("width", [3, 8, 20, 50])
@pytest.mark.parametrize("height", [1, 2, 9, 21, 51])
def test_ellipse_crosses_approximation(width, height):
    shape = (width * 2 + 1, height * 2 + 1)
    expected = footprint_ellipse(shape)
    decomposed = cross_decompose_footprint(expected)
    approximate = footprint_from_sequence(decomposed)
    assert approximate.shape == expected.shape

    # verify that maximum error does not exceed some fraction of the size
    error = np.sum(np.abs(expected.astype(int) - approximate.astype(int)))
    max_error = 0.01
    assert error / expected.size <= max_error


def test_cross_decompose_footprint_asymmetric():
    asymmetric = np.ones((3, 3), dtype=bool)
    asymmetric[0, :] = 0  # Still concave
    with pytest.raises(ValueError, match=r"footprint is not symmetric"):
        cross_decompose_footprint(asymmetric)


def test_cross_decompose_footprint_concave():
    concave = np.ones((3, 3), dtype=bool)
    concave[0, 1] = 0
    concave[-1, 1] = 0  # Still symmetric
    with pytest.raises(ValueError, match=r"footprint is not convex"):
        cross_decompose_footprint(concave)


def test_cross_decompose_footprint_even():
    even = np.ones((4, 3), dtype=bool)
    with pytest.raises(ValueError, match=r"footprint is not of uneven length"):
        cross_decompose_footprint(even)


def test_disk_series_approximation_unavailable():
    # ValueError if radius is too large (only precomputed up to radius=250)
    with pytest.raises(ValueError):
        footprints.disk(radius=10000, decomposition="sequence")


def test_ball_series_approximation_unavailable():
    # ValueError if radius is too large (only precomputed up to radius=100)
    with pytest.raises(ValueError):
        footprints.ball(radius=10000, decomposition="sequence")


# skimage.morphology.mirror_footprint --------------------------------------------------


@pytest.mark.parametrize("as_sequence", [tuple, None])
def test_mirror_footprint(as_sequence):
    footprint = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], np.uint8)
    expected_res = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    if as_sequence is not None:
        footprint = as_sequence([(footprint, 2), (footprint.T, 3)])
        expected_res = as_sequence([(expected_res, 2), (expected_res.T, 3)])

    actual_res = mirror_footprint(footprint)
    assert type(expected_res) is type(actual_res)
    assert_equal(expected_res, actual_res)


# skimage.morphology.pad_footprint -----------------------------------------------------


@pytest.mark.parametrize("as_sequence", [tuple, None])
@pytest.mark.parametrize("pad_end", [True, False])
def test_pad_footprint(as_sequence, pad_end):
    footprint = np.array([[0, 0], [1, 0], [1, 1]], np.uint8)
    pad_width = [(0, 0), (0, 1)] if pad_end is True else [(0, 0), (1, 0)]
    expected_res = np.pad(footprint, pad_width)
    if as_sequence is not None:
        footprint = as_sequence([(footprint, 2), (footprint.T, 3)])
        expected_res = as_sequence([(expected_res, 2), (expected_res.T, 3)])

    actual_res = pad_footprint(footprint, pad_end=pad_end)
    assert type(expected_res) is type(actual_res)
    assert_equal(expected_res, actual_res)


class Test_footprint_rectangule:
    @pytest.mark.parametrize("i", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("j", [0, 1, 2, 3, 4])
    def test_rectangle(self, i, j):
        desired = np.ones((i, j), dtype='uint8')
        actual = footprint_rectangle((i, j))
        assert_equal(actual, desired)

    @pytest.mark.parametrize("i", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("j", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
    def test_cuboid(self, i, j, k):
        desired = np.ones((i, j, k), dtype='uint8')
        actual = footprint_rectangle((i, j, k))
        assert_equal(actual, desired)

    @pytest.mark.parametrize("shape", [(3,), (5, 5), (5, 5, 7)])
    @pytest.mark.parametrize("decomposition", ["separable", "sequence"])
    def test_decomposition(self, shape, decomposition):
        regular = footprint_rectangle(shape)
        decomposed = footprint_rectangle(shape, decomposition=decomposition)
        recomposed = footprint_from_sequence(decomposed)
        assert_equal(recomposed, regular)

    @pytest.mark.parametrize("shape", [(2,), (3, 4)])
    def test_uneven_sequence_decomposition_warning(self, shape):
        """Should fall back to decomposition="separable" for uneven footprint size."""
        desired = footprint_rectangle(shape, decomposition="separable")
        regex = "decomposition='sequence' is only supported for uneven footprints"
        with pytest.warns(UserWarning, match=regex) as record:
            actual = footprint_rectangle(shape, decomposition="sequence")
        assert_stacklevel(record)
        assert_equal(actual, desired)
