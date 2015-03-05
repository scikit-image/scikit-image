import numpy as np
from numpy.testing import (assert_array_equal, assert_equal, assert_raises,
                           assert_warns)
from skimage.morphology import remove_small_objects

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
    assert_warns(UserWarning, remove_small_objects, image, min_size=6)


def test_float_input():
    float_test = np.random.rand(5, 5)
    assert_raises(TypeError, remove_small_objects, float_test)


def test_negative_input():
    negative_int = np.random.randint(-4, -1, size=(5, 5))
    assert_raises(ValueError, remove_small_objects, negative_int)


if __name__ == "__main__":
    np.testing.run_module_suite()
