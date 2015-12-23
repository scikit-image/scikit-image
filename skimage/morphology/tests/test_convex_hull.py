import numpy as np
from numpy.testing import assert_array_equal, assert_raises
from numpy.testing.decorators import skipif
from skimage.morphology import convex_hull_image, convex_hull_object
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

    # Test that an error is raised on passing a 3D image:
    image3d = np.empty((5, 5, 5))
    assert_raises(ValueError, convex_hull_image, image3d)


@skipif(not scipy_spatial)
def test_qhull_offset_example():
    nonzeros = (([1367, 1368, 1368, 1368, 1369, 1369, 1369, 1369, 1369, 1370,
                  1370, 1370, 1370, 1370, 1370, 1370, 1371, 1371, 1371, 1371,
                  1371, 1371, 1371, 1371, 1371, 1372, 1372, 1372, 1372, 1372,
                  1372, 1372, 1372, 1372, 1373, 1373, 1373, 1373, 1373, 1373,
                  1373, 1373, 1373, 1374, 1374, 1374, 1374, 1374, 1374, 1374,
                  1375, 1375, 1375, 1375, 1375, 1376, 1376, 1376, 1377]),
                ([151, 150, 151, 152, 149, 150, 151, 152, 153, 148, 149, 150,
                 151, 152, 153, 154, 147, 148, 149, 150, 151, 152, 153, 154,
                 155, 146, 147, 148, 149, 150, 151, 152, 153, 154, 146, 147,
                 148, 149, 150, 151, 152, 153, 154, 147, 148, 149, 150, 151,
                 152, 153, 148, 149, 150, 151, 152, 149, 150, 151, 150]))
    image = np.zeros((1392, 1040), dtype=bool)
    image[nonzeros] = True
    expected = image.copy()
    assert_array_equal(convex_hull_image(image), expected)


@skipif(not scipy_spatial)
def test_pathological_qhull_example():
    image = np.array(
                [[0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0]], dtype=bool)
    expected = np.array(
                [[0, 0, 0, 1, 1, 1, 0],
                 [0, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 0, 0, 0]], dtype=bool)
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


@skipif(not scipy_spatial)
def test_object():
    image = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 0, 0, 1, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 1, 0],
         [1, 0, 0, 0, 0, 0, 1, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

    expected4 = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 0, 0, 1, 0, 1],
         [1, 1, 1, 0, 0, 0, 0, 1, 0],
         [1, 1, 0, 0, 0, 0, 1, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

    assert_array_equal(convex_hull_object(image, 4), expected4)

    expected8 = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 0, 0, 1, 1, 1],
         [1, 1, 1, 0, 0, 0, 1, 1, 1],
         [1, 1, 0, 0, 0, 0, 1, 1, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

    assert_array_equal(convex_hull_object(image, 8), expected8)

    assert_raises(ValueError, convex_hull_object, image, 7)

    # Test that an error is raised on passing a 3D image:
    image3d = np.empty((5, 5, 5))
    assert_raises(ValueError, convex_hull_object, image3d)

if __name__ == "__main__":
    np.testing.run_module_suite()
