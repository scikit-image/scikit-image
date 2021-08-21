import numpy as np
from skimage.measure import find_contours

from skimage._shared import testing
from skimage._shared.testing import assert_array_equal
from pytest import raises


a = np.ones((8, 8), dtype=np.float32)
a[1:-1, 1] = 0
a[1, 1:-1] = 0

x, y = np.mgrid[-1:1:5j, -1:1:5j]
r = np.sqrt(x**2 + y**2)


def test_binary():
    ref = [[6. ,  1.5],
           [5. ,  1.5],
           [4. ,  1.5],
           [3. ,  1.5],
           [2. ,  1.5],
           [1.5,  2. ],
           [1.5,  3. ],
           [1.5,  4. ],
           [1.5,  5. ],
           [1.5,  6. ],
           [1. ,  6.5],
           [0.5,  6. ],
           [0.5,  5. ],
           [0.5,  4. ],
           [0.5,  3. ],
           [0.5,  2. ],
           [0.5,  1. ],
           [1. ,  0.5],
           [2. ,  0.5],
           [3. ,  0.5],
           [4. ,  0.5],
           [5. ,  0.5],
           [6. ,  0.5],
           [6.5,  1. ],
           [6. ,  1.5]]

    contours = find_contours(a, 0.5, positive_orientation='high')
    assert len(contours) == 1
    assert_array_equal(contours[0][::-1], ref)


# target contour for mask tests
mask_contour = [
    [6. ,  0.5],
    [5. ,  0.5],
    [4. ,  0.5],
    [3. ,  0.5],
    [2. ,  0.5],
    [1. ,  0.5],
    [0.5,  1. ],
    [0.5,  2. ],
    [0.5,  3. ],
    [0.5,  4. ],
    [0.5,  5. ],
    [0.5,  6. ],
    [1. ,  6.5],
    [1.5,  6. ],
    [1.5,  5. ],
    [1.5,  4. ],
    [1.5,  3. ],
    [1.5,  2. ],
    [2. ,  1.5],
    [3. ,  1.5],
    [4. ,  1.5],
    [5. ,  1.5],
    [6. ,  1.5],
]

mask = np.ones((8, 8), dtype=bool)
# Some missing data that should result in a hole in the contour:
mask[7, 0:3] = False


def test_nodata():
    # Test missing data via NaNs in input array
    b = np.copy(a)
    b[~mask] = np.nan
    contours = find_contours(b, 0.5, positive_orientation='high')
    assert len(contours) == 1
    assert_array_equal(contours[0], mask_contour)


def test_mask():
    # Test missing data via explicit masking
    contours = find_contours(a, 0.5, positive_orientation='high', mask=mask)
    assert len(contours) == 1
    assert_array_equal(contours[0], mask_contour)


def test_mask_shape():
    bad_mask = np.ones((8, 7), dtype=bool)
    with raises(ValueError, match='shape'):
        find_contours(a, 0, mask=bad_mask)


def test_mask_dtype():
    bad_mask = np.ones((8, 8), dtype=np.uint8)
    with raises(TypeError, match='binary'):
        find_contours(a, 0, mask=bad_mask)


def test_float():
    contours = find_contours(r, 0.5)
    assert len(contours) == 1
    assert_array_equal(contours[0],
                    [[ 2.,  3.],
                     [ 1.,  2.],
                     [ 2.,  1.],
                     [ 3.,  2.],
                     [ 2.,  3.]])


def test_memory_order():
    contours = find_contours(np.ascontiguousarray(r), 0.5)
    assert len(contours) == 1

    contours = find_contours(np.asfortranarray(r), 0.5)
    assert len(contours) == 1


def test_invalid_input():
    with testing.raises(ValueError):
        find_contours(r, 0.5, 'foo', 'bar')
    with testing.raises(ValueError):
        find_contours(r[..., None], 0.5)


def test_nodata_levelNone():
    # Test missing data via NaNs in input array
    b = np.copy(a)
    b[~mask] = np.nan
    contours = find_contours(b, level=None, positive_orientation='high')
    assert len(contours) == 1
    assert_array_equal(contours[0], mask_contour)


def test_mask_levelNone():
    # Test missing data via explicit masking
    contours = find_contours(a, level=None, positive_orientation='high',
                             mask=mask)
    assert len(contours) == 1
    assert_array_equal(contours[0], mask_contour)


def test_mask_shape_levelNone():
    bad_mask = np.ones((8, 7), dtype=bool)
    with raises(ValueError, match='shape'):
        find_contours(a, level=None, mask=bad_mask)


def test_mask_dtype_levelNone():
    bad_mask = np.ones((8, 8), dtype=np.uint8)
    with raises(TypeError, match='binary'):
        find_contours(a, level=None, mask=bad_mask)


def test_memory_order_levelNone():
    contours = find_contours(np.ascontiguousarray(r), level=None)
    assert len(contours) == 1

    contours = find_contours(np.asfortranarray(r), level=None)
    assert len(contours) == 1


def test_level_default():
    # image with range [0.9, 0.91]
    image = np.random.random((100, 100)) * 0.01 + 0.9
    contours = find_contours(image)  # use default level
    # many contours should be found
    assert len(contours) > 1
