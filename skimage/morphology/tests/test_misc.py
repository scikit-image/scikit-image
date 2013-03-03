import numpy as np
from numpy.testing import assert_array_equal, assert_equal
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
    observed = remove_small_objects(test_image, min_size=7,
                                                            connectivity=2)
    assert_array_equal(observed, expected)

def test_in_place():
    observed = remove_small_objects(test_image, min_size=6,
                                                            in_place=True)
    assert_equal(observed is test_image, True, 
        "remove_small_objects in_place argument failed.")

if __name__ == "__main__":
    np.testing.run_module_suite()
