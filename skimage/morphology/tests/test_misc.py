import numpy as np
import pytest
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    remove_close_objects,
)

from skimage._shared import testing
from skimage._shared.testing import assert_array_equal, assert_equal
from skimage._shared._warnings import expected_warnings


test_image = np.array([[0, 0, 0, 1, 0],
                       [1, 1, 1, 0, 0],
                       [1, 1, 1, 0, 1]], bool)


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
    observed = remove_small_objects(test_image, min_size=6, in_place=True)
    assert_equal(observed is test_image, True,
                 "remove_small_objects in_place argument failed.")


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
                             [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]], np.bool_)


def test_one_connectivity_holes():
    expected = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]], np.bool_)
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
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]], np.bool_)
    observed = remove_small_holes(test_holes_image, area_threshold=3,
                                  connectivity=2)
    assert_array_equal(observed, expected)


def test_in_place_holes():
    observed = remove_small_holes(test_holes_image, area_threshold=3,
                                  in_place=True)
    assert_equal(observed is test_holes_image, True,
                 "remove_small_holes in_place argument failed.")


def test_labeled_image_holes():
    labeled_holes_image = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                                    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
                                    [0, 0, 0, 0, 0, 0, 0, 2, 0, 2],
                                    [0, 0, 0, 0, 0, 0, 0, 2, 2, 2]],
                                   dtype=np.int_)
    expected = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]], dtype=np.bool_)
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
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]], dtype=np.bool_)
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
                                   dtype=np.int_)
    with expected_warnings(['use a boolean array?']):
        remove_small_holes(labeled_holes_image, area_threshold=3)
    remove_small_holes(labeled_holes_image.astype(bool), area_threshold=3)


def test_float_input_holes():
    float_test = np.random.rand(5, 5)
    with testing.raises(TypeError):
        remove_small_holes(float_test)


class TestRemoveCloseObjects:

    @pytest.mark.parametrize("minimal_distance", [10, 20, 30, 49])
    def test_linspace_1d(self, minimal_distance):
        max_step = 50
        offset = np.linspace(1, max_step, max_step, dtype=np.intp)[::-1]
        positions = np.cumsum(offset)
        image = np.zeros(positions.max() + 2, dtype=bool)
        image[positions] = 1

        result = remove_close_objects(image, minimal_distance=minimal_distance)

        diff = np.diff(np.nonzero(result)[0])
        assert diff.min() == minimal_distance + 1

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
            dtype=np.uint8
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
            dtype=bool
        )

        image = priority.astype(bool)
        result = remove_close_objects(
            image, minimal_distance=3, priority=priority
        )
        assert_array_equal(result, desired)

    @pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5])
    def test_large_objects_nd(self, ndim):
        shape = (5,) * ndim
        a = np.ones(shape, dtype=np.uint8)
        a[2, ...] = 0
        desired = a.astype(bool)
        desired[2:, ...] = False
        image = a.astype(bool)

        result = remove_close_objects(image, minimal_distance=2)
        assert_array_equal(result, desired)

    @pytest.mark.parametrize("value", [True, False])
    def test_constant(self, value):
        image = np.empty((10, 10), dtype=bool)
        image.fill(value)

        result = remove_close_objects(image, 3)
        assert_array_equal(image, result)

    def test_empty(self):
        image = np.empty((3, 3, 0), dtype=np.bool_)
        result = remove_close_objects(image, 3)
        assert_equal(image, result)

    def test_priority(self):
        image = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=bool)

        # Default priority is row-major (C-style) order
        result = remove_close_objects(image, 3)
        desired = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=bool)
        assert_array_equal(result, desired)

        # But given a priority that order can be overuled
        priority = np.array([[0, 0, 1], [0, 0, 0], [2, 0, 0]], dtype=int)
        result = remove_close_objects(
            image, minimal_distance=3, priority=priority
        )
        desired = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=bool)
        assert_array_equal(result, desired)

    @pytest.mark.parametrize("dtype", [np.uint8, np.int8])
    def test_view_on_byte_sized(self, dtype):
        """Test behavior if image is a view on 1 byte sized numeric dtypes."""
        image = np.array([-2, 0, 2], dtype=dtype)

        # When using a view, object values don't change
        result_view = remove_close_objects(image.view(bool), 2)
        desired_view = np.array([-2, 0, 0], dtype=dtype)
        assert result_view.dtype is np.dtype(bool)
        assert_array_equal(result_view.view(dtype), desired_view)

        # When using astype, the object values > 0 are replaced with 1
        result_astype = remove_close_objects(image.astype(bool), 2)
        desired_astype = np.array([1, 0, 0], dtype=np.uint8)
        assert_array_equal(result_astype, result_view)
        assert_array_equal(result_astype.view(np.uint8), desired_astype)
