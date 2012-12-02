import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing.decorators import skipif
from skimage.morphology import convex_hull_image
from skimage.morphology._convex_hull import possible_hull

try:
    import scipy.spatial
    scipy_spatial = True
except ImportError:
    scipy_spatial = False


@skipif(not scipy_spatial)
def test_basic():
    image = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

    expected = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

    assert_array_equal(convex_hull_image(image), expected)


@skipif(not scipy_spatial)
def test_possible_hull():
    image = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

    expected = np.array([[1, 4],
                         [2, 3],
                         [3, 2],
                         [4, 1],
                         [4, 1],
                         [3, 2],
                         [2, 3],
                         [1, 4],
                         [2, 5],
                         [3, 6],
                         [4, 7],
                         [2, 5],
                         [3, 6],
                         [4, 7],
                         [4, 2],
                         [4, 3],
                         [4, 4],
                         [4, 5],
                         [4, 6]])

    ph = possible_hull(image)
    assert_array_equal(ph, expected)

if __name__ == "__main__":
    np.testing.run_module_suite()
