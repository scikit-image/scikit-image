import numpy as np
import pytest
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    remove_near_objects,
)

from skimage._shared import testing
from skimage._shared.testing import assert_array_equal, assert_equal
from skimage._shared._warnings import expected_warnings


test_image = np.array([[0, 0, 0, 1, 0],
                       [1, 1, 1, 0, 0],
                       [1, 1, 1, 0, 1]], bool)

# Dtypes supported by remove_near_objects
supported_dtypes = [
    bool,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
]


def test_one_connectivity():
    expected = np.array([[0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0],
                         [1, 1, 1, 0, 0]], bool)
    observed = remove_small_objects(test_image, min_size=6)
    assert_array_equal(observed, expected)


def test_two_connectivity():
    expected = np.array([[0, 0, 0, 1, 0],
                         [1, 1, 1, 0, 0],
                         [1, 1, 1, 0, 0]], bool)
    observed = remove_small_objects(test_image, min_size=7, connectivity=2)
    assert_array_equal(observed, expected)


def test_in_place():
    image = test_image.copy()
    with expected_warnings(["in_place argument is deprecated"]):
        observed = remove_small_objects(image, min_size=6, in_place=True)
    assert_equal(observed is image, True,
                 "remove_small_objects in_place argument failed.")


@pytest.mark.parametrize("in_dtype", [bool, int, np.int32])
@pytest.mark.parametrize("out_dtype", [bool, int, np.int32])
def test_out(in_dtype, out_dtype):
    image = test_image.astype(in_dtype, copy=True)
    expected_out = np.empty_like(test_image, dtype=out_dtype)

    if out_dtype != bool:
        # object with only 1 label will warn on non-bool output dtype
        exp_warn = ["Only one label was provided"]
    else:
        exp_warn = []

    with expected_warnings(exp_warn):
        out = remove_small_objects(image, min_size=6, out=expected_out)

    assert out is expected_out


def test_labeled_image():
    labeled_image = np.array([[2, 2, 2, 0, 1],
                              [2, 2, 2, 0, 1],
                              [2, 0, 0, 0, 0],
                              [0, 0, 3, 3, 3]], dtype=int)
    expected = np.array([[2, 2, 2, 0, 0],
                         [2, 2, 2, 0, 0],
                         [2, 0, 0, 0, 0],
                         [0, 0, 3, 3, 3]], dtype=int)
    observed = remove_small_objects(labeled_image, min_size=3)
    assert_array_equal(observed, expected)


def test_uint_image():
    labeled_image = np.array([[2, 2, 2, 0, 1],
                              [2, 2, 2, 0, 1],
                              [2, 0, 0, 0, 0],
                              [0, 0, 3, 3, 3]], dtype=np.uint8)
    expected = np.array([[2, 2, 2, 0, 0],
                         [2, 2, 2, 0, 0],
                         [2, 0, 0, 0, 0],
                         [0, 0, 3, 3, 3]], dtype=np.uint8)
    observed = remove_small_objects(labeled_image, min_size=3)
    assert_array_equal(observed, expected)


def test_single_label_warning():
    image = np.array([[0, 0, 0, 1, 0],
                      [1, 1, 1, 0, 0],
                      [1, 1, 1, 0, 0]], int)
    with expected_warnings(['use a boolean array?']):
        remove_small_objects(image, min_size=6)


def test_float_input():
    float_test = np.random.rand(5, 5)
    with testing.raises(TypeError):
        remove_small_objects(float_test)


def test_negative_input():
    negative_int = np.random.randint(-4, -1, size=(5, 5))
    with testing.raises(ValueError):
        remove_small_objects(negative_int)


test_holes_image = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                             [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                             [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]], bool)


def test_one_connectivity_holes():
    expected = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]], bool)
    observed = remove_small_holes(test_holes_image, area_threshold=3)
    assert_array_equal(observed, expected)


def test_two_connectivity_holes():
    expected = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]], bool)
    observed = remove_small_holes(test_holes_image, area_threshold=3,
                                  connectivity=2)
    assert_array_equal(observed, expected)


def test_in_place_holes():
    image = test_holes_image.copy()
    with expected_warnings(["in_place argument is deprecated"]):
        observed = remove_small_holes(image, area_threshold=3, in_place=True)
    assert_equal(observed is image, True,
                 "remove_small_holes in_place argument failed.")


def test_out_remove_small_holes():
    image = test_holes_image.copy()
    expected_out = np.empty_like(image)
    out = remove_small_holes(image, area_threshold=3, out=expected_out)

    assert out is expected_out


def test_non_bool_out():
    image = test_holes_image.copy()
    expected_out = np.empty_like(image, dtype=int)
    with testing.raises(TypeError):
        remove_small_holes(image, area_threshold=3, out=expected_out)


def test_labeled_image_holes():
    labeled_holes_image = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
                                    [0, 0, 0, 0, 0, 0, 0, 2, 0, 2],
                                    [0, 0, 0, 0, 0, 0, 0, 2, 2, 2]],
                                   dtype=int)
    expected = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]], dtype=bool)
    with expected_warnings(['returned as a boolean array']):
        observed = remove_small_holes(labeled_holes_image, area_threshold=3)
    assert_array_equal(observed, expected)


def test_uint_image_holes():
    labeled_holes_image = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
                                    [0, 0, 0, 0, 0, 0, 0, 2, 0, 2],
                                    [0, 0, 0, 0, 0, 0, 0, 2, 2, 2]],
                                   dtype=np.uint8)
    expected = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]], dtype=bool)
    with expected_warnings(['returned as a boolean array']):
        observed = remove_small_holes(labeled_holes_image, area_threshold=3)
    assert_array_equal(observed, expected)


def test_label_warning_holes():
    labeled_holes_image = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
                                    [0, 0, 0, 0, 0, 0, 0, 2, 0, 2],
                                    [0, 0, 0, 0, 0, 0, 0, 2, 2, 2]],
                                   dtype=int)
    with expected_warnings(['use a boolean array?']):
        remove_small_holes(labeled_holes_image, area_threshold=3)
    remove_small_holes(labeled_holes_image.astype(bool), area_threshold=3)


def test_float_input_holes():
    float_test = np.random.rand(5, 5)
    with testing.raises(TypeError):
        remove_small_holes(float_test)


class Test_remove_near_objects:

    @pytest.mark.parametrize("minimal_distance", [2.1, 5, 30.99, 49])
    @pytest.mark.parametrize("dtype", supported_dtypes)
    def test_minimal_distance_1d(self, minimal_distance, dtype):
        # First 3 objects are only just to close, last one is just far enough
        d = int(np.floor(minimal_distance))
        image = np.zeros(d * 3 + 2, dtype=dtype)
        image[[0, d, 2 * d, 3 * d + 1]] = -1  # Negative values are objects
        desired = np.zeros_like(image, dtype=dtype)
        desired[[0, 2 * d, 3 * d + 1]] = -1

        result = remove_near_objects(image, minimal_distance=minimal_distance)
        assert result.dtype == desired.dtype
        assert_array_equal(result, desired)

    def test_handcrafted_2d(self):
        priority = np.array(
            [[8, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9],
             [8, 8, 8, 0, 0, 0, 0, 0, 0, 9, 9],
             [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 0],
             [2, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7]],
            dtype=np.uint8,
        )
        desired = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
             [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]],
            dtype=bool,
        )

        image = priority.astype(bool)
        result = remove_near_objects(image, minimal_distance=3, priority=priority)
        assert_array_equal(result, desired)

    @pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5])
    def test_large_objects_nd(self, ndim):
        shape = (5,) * ndim
        a = np.ones(shape, dtype=np.uint8)
        a[-2, ...] = 0
        desired = a.astype(bool)
        desired[-2:, ...] = False
        image = a.astype(bool)

        result = remove_near_objects(image, minimal_distance=2)
        assert_array_equal(result, desired)

    @pytest.mark.parametrize("value", [True, False])
    def test_constant(self, value):
        image = np.empty((10, 10), dtype=bool)
        image.fill(value)

        result = remove_near_objects(image, minimal_distance=3)
        assert_array_equal(image, result)

    def test_empty(self):
        image = np.empty((3, 3, 0), dtype=np.bool_)
        result = remove_near_objects(image, minimal_distance=3)
        assert_equal(image, result)

    def test_priority(self):
        image = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=bool)

        # Default priority is reverse row-major (C-style) order
        result = remove_near_objects(image, minimal_distance=3)
        desired = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=bool)
        assert_array_equal(result, desired)

        # Assigning priority with equal values shows same order
        priority = np.array([[0, 0, 2], [0, 0, 0], [2, 0, 0]], dtype=int)
        result = remove_near_objects(image, minimal_distance=3, priority=priority)
        desired = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=bool)
        assert_array_equal(result, desired)

        # But given a priority that order can be overuled
        priority = np.array([[0, 0, 2], [0, 0, 0], [1, 0, 0]], dtype=int)
        result = remove_near_objects(image, minimal_distance=3, priority=priority)
        desired = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=bool)
        assert_array_equal(result, desired)

    def test_out(self):
        image = np.array([True, False, True])
        image_copy = image.copy()
        desired = np.array([False, False, True], dtype=bool)

        # By default, input image is not modified
        remove_near_objects(image, minimal_distance=2)
        assert_array_equal(image, image_copy)

        remove_near_objects(image, minimal_distance=2, out=image)
        assert_array_equal(image, desired)


    @pytest.mark.parametrize("minimal_distance", [-10, -0.1])
    def test_negative_minimal_distance(self, minimal_distance):
        image = np.array([True, False, True])
        with pytest.raises(ValueError, match="must be >= 0"):
            remove_near_objects(image, minimal_distance=minimal_distance)

    def test_non_zero(self):
        # Check an object with different values is recognized as one as long
        # as they aren't zero
        image = np.array([1, 0, 0, -1, 2, 99, -10])
        desired = np.array([1, 0, 0, 0, 0, 0, 0])
        priority = np.arange(image.size)[::-1]
        result = remove_near_objects(image, minimal_distance=3, priority=priority)
        assert_array_equal(result, desired)

    def test_diagonal_selem(self):
        # Given a constant image and a diagonal structuring element, two
        # objects are detected similar to black and white on a checkerboard
        footprint = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool)
        image = np.ones((100, 50))
        desired = np.array([[1, 0] * 25, [0, 1] * 25] * 50, dtype=image.dtype)

        result = remove_near_objects(image, minimal_distance=1, footprint=footprint)
        assert_array_equal(result, desired)

    def test_nan(self):
        # Check that NaNs are treated as objects
        image = np.array([np.nan, np.nan, np.nan, 0, np.nan, np.nan, np.nan])
        desired = np.array([0, 0, 0, 0, np.nan, np.nan, np.nan])
        result = remove_near_objects(image, minimal_distance=2)
        assert_array_equal(result, desired)

    def test_p_norm(self):
        image = np.array([[2, 0], [0, 1]])
        removed = np.array([[2, 0], [0, 0]])

        # p_norm=2, default (Euclidean distance)
        result = remove_near_objects(
            image, minimal_distance=1.4, connectivity=1, priority=image
        )
        assert_array_equal(result, image)
        result = remove_near_objects(
            image, minimal_distance=np.sqrt(2), connectivity=1, priority=image
        )
        assert_array_equal(result, removed)

        # p_norm=1 (Manhatten distance)
        result = remove_near_objects(
            image, minimal_distance=1.9, p_norm=1, connectivity=1, priority=image
        )
        assert_array_equal(result, image)
        result = remove_near_objects(
            image, minimal_distance=2, p_norm=1, connectivity=1, priority=image
        )
        assert_array_equal(result, removed)

        # p_norm=np.inf (Chebyshev distance)
        result = remove_near_objects(
            image,
            minimal_distance=0.9,
            p_norm=np.inf,
            connectivity=1,
            priority=image,
        )
        assert_array_equal(result, image)
        result = remove_near_objects(
            image, minimal_distance=1, p_norm=np.inf, connectivity=1, priority=image
        )
        assert_array_equal(result, removed)

    def test_wrong_priority_shape(self):
        image = np.zeros((10, 10))
        priority = np.ones((10, 9))
        with pytest.raises(ValueError, match="priority.*shape"):
            remove_near_objects(image, minimal_distance=3, priority=priority)
