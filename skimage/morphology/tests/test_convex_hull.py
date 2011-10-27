import numpy as np
from numpy.testing import assert_array_equal
from skimage.morphology import convex_hull

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

    assert_array_equal(convex_hull(image), expected)

if __name__ == "__main__":
    np.testing.run_module_suite()
