import numpy as np

from scikits.image.filter import median_filter


def test_00_00_zeros():
    '''The median filter on an array of all zeros should be zero'''
    result = median_filter(np.zeros((10, 10)), np.ones((10, 10), bool), 3)
    assert np.all(result == 0)


def test_00_01_all_masked():
    '''Test a completely masked image

    Regression test of IMG-1029'''
    result = median_filter(np.zeros((10, 10)), np.zeros((10, 10), bool), 3)
    assert (np.all(result == 0))


def test_00_02_all_but_one_masked():
    mask = np.zeros((10, 10), bool)
    mask[5, 5] = True
    median_filter(np.zeros((10, 10)), mask, 3)


def test_01_01_mask():
    '''The median filter, masking a single value'''
    img = np.zeros((10, 10))
    img[5, 5] = 1
    mask = np.ones((10, 10), bool)
    mask[5, 5] = False
    result = median_filter(img, mask, 3)
    assert (np.all(result[mask] == 0))
    np.testing.assert_equal(result[5, 5], 1)


def test_02_01_median():
    '''A median filter larger than the image = median of image'''
    np.random.seed(0)
    img = np.random.uniform(size=(9, 9))
    result = median_filter(img, np.ones((9, 9), bool), 20)
    np.testing.assert_equal(result[0, 0], np.median(img))
    assert (np.all(result == np.median(img)))


def test_02_02_median_bigger():
    '''Use an image of more than 255 values to test approximation'''
    np.random.seed(0)
    img = np.random.uniform(size=(20, 20))
    result = median_filter(img, np.ones((20, 20), bool), 40)
    sorted = np.ravel(img)
    sorted.sort()
    min_acceptable = sorted[198]
    max_acceptable = sorted[202]
    assert (np.all(result >= min_acceptable))
    assert (np.all(result <= max_acceptable))


def test_03_01_shape():
    '''Make sure the median filter is the expected octagonal shape'''

    radius = 5
    a_2 = int(radius / 2.414213)
    i, j = np.mgrid[-10:11, -10:11]
    octagon = np.ones((21, 21), bool)
    #
    # constrain the octagon mask to be the points that are on
    # the correct side of the 8 edges
    #
    octagon[i < -radius] = False
    octagon[i > radius] = False
    octagon[j < -radius] = False
    octagon[j > radius] = False
    octagon[i + j < -radius - a_2] = False
    octagon[j - i > radius + a_2] = False
    octagon[i + j > radius + a_2] = False
    octagon[i - j > radius + a_2] = False
    np.random.seed(0)
    img = np.random.uniform(size=(21, 21))
    result = median_filter(img, np.ones((21, 21), bool), radius)
    sorted = img[octagon]
    sorted.sort()
    min_acceptable = sorted[len(sorted) / 2 - 1]
    max_acceptable = sorted[len(sorted) / 2 + 1]
    assert (result[10, 10] >= min_acceptable)
    assert (result[10, 10] <= max_acceptable)


def test_04_01_half_masked():
    '''Make sure that the median filter can handle large masked areas.'''
    img = np.ones((20, 20))
    mask = np.ones((20, 20), bool)
    mask[10:, :] = False
    img[~ mask] = 2
    img[1, 1] = 0  # to prevent short circuit for uniform data.
    result = median_filter(img, mask, 5)
    # in partial coverage areas, the result should be only
    # from the masked pixels
    assert (np.all(result[:14, :] == 1))
    # in zero coverage areas, the result should be the lowest
    # value in the valid area
    assert (np.all(result[15:, :] == np.min(img[mask])))


def test_default_values():
    img = (np.random.random((20, 20)) * 255).astype(np.uint8)
    mask = np.ones((20, 20), dtype=np.uint8)
    result1 = median_filter(img, mask, radius=2, percent=50)
    result2 = median_filter(img)
    np.testing.assert_array_equal(result1, result2)


if __name__ == "__main__":
    np.testing.run_module_suite()
