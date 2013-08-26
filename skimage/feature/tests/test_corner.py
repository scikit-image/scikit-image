import numpy as np
from numpy.testing import (assert_array_equal, assert_raises,
                           assert_almost_equal)

from skimage import data
from skimage import img_as_float
from skimage.color import rgb2gray

from skimage.feature import (corner_moravec, corner_harris, corner_shi_tomasi,
                             corner_subpix, peak_local_max, corner_peaks,
                             corner_kitchen_rosenfeld, corner_foerstner,
                             corner_fast, corner_fast_orientation)


def test_square_image():
    im = np.zeros((50, 50)).astype(float)
    im[:25, :25] = 1.

    # Moravec
    results = peak_local_max(corner_moravec(im))
    # interest points along edge
    assert len(results) == 57

    # Harris
    results = peak_local_max(corner_harris(im))
    # interest at corner
    assert len(results) == 1

    # Shi-Tomasi
    results = peak_local_max(corner_shi_tomasi(im))
    # interest at corner
    assert len(results) == 1


def test_noisy_square_image():
    im = np.zeros((50, 50)).astype(float)
    im[:25, :25] = 1.
    np.random.seed(seed=1234)
    im = im + np.random.uniform(size=im.shape) * .2

    # Moravec
    results = peak_local_max(corner_moravec(im))
    # undefined number of interest points
    assert results.any()

    # Harris
    results = peak_local_max(corner_harris(im, sigma=1.5))
    assert len(results) == 1

    # Shi-Tomasi
    results = peak_local_max(corner_shi_tomasi(im, sigma=1.5))
    assert len(results) == 1


def test_squared_dot():
    im = np.zeros((50, 50))
    im[4:8, 4:8] = 1
    im = img_as_float(im)

    # Moravec fails

    # Harris
    results = peak_local_max(corner_harris(im))
    assert (results == np.array([[6, 6]])).all()

    # Shi-Tomasi
    results = peak_local_max(corner_shi_tomasi(im))
    assert (results == np.array([[6, 6]])).all()


def test_rotated_lena():
    """
    The harris filter should yield the same results with an image and it's
    rotation.
    """
    im = img_as_float(data.lena().mean(axis=2))
    im_rotated = im.T

    # Moravec
    results = peak_local_max(corner_moravec(im))
    results_rotated = peak_local_max(corner_moravec(im_rotated))
    assert (np.sort(results[:, 0]) == np.sort(results_rotated[:, 1])).all()
    assert (np.sort(results[:, 1]) == np.sort(results_rotated[:, 0])).all()

    # Harris
    results = peak_local_max(corner_harris(im))
    results_rotated = peak_local_max(corner_harris(im_rotated))
    assert (np.sort(results[:, 0]) == np.sort(results_rotated[:, 1])).all()
    assert (np.sort(results[:, 1]) == np.sort(results_rotated[:, 0])).all()

    # Shi-Tomasi
    results = peak_local_max(corner_shi_tomasi(im))
    results_rotated = peak_local_max(corner_shi_tomasi(im_rotated))
    assert (np.sort(results[:, 0]) == np.sort(results_rotated[:, 1])).all()
    assert (np.sort(results[:, 1]) == np.sort(results_rotated[:, 0])).all()


def test_subpix():
    img = np.zeros((50, 50))
    img[:25, :25] = 255
    img[25:, 25:] = 255
    corner = peak_local_max(corner_harris(img), num_peaks=1)
    subpix = corner_subpix(img, corner)
    assert_array_equal(subpix[0], (24.5, 24.5))


def test_subpix_border():
    img = np.zeros((50, 50))
    img[1:25,1:25] = 255
    img[25:-1,25:-1] = 255
    corner = corner_peaks(corner_harris(img), min_distance=1)
    subpix = corner_subpix(img, corner, window_size=11)
    ref = np.array([[ 0.52040816,  0.52040816],
                    [ 0.52040816, 24.47959184],
                    [24.47959184,  0.52040816],
                    [24.5       , 24.5       ],
                    [24.52040816, 48.47959184],
                    [48.47959184, 24.52040816],
                    [48.47959184, 48.47959184]])
    assert_almost_equal(subpix, ref)


def test_num_peaks():
    """For a bunch of different values of num_peaks, check that
    peak_local_max returns exactly the right amount of peaks. Test
    is run on Lena in order to produce a sufficient number of corners"""

    lena_corners = corner_harris(data.lena())

    for i in range(20):
        n = np.random.random_integers(20)
        results = peak_local_max(lena_corners, num_peaks=n)
        assert (results.shape[0] == n)


def test_corner_peaks():
    response = np.zeros((5, 5))
    response[2:4, 2:4] = 1

    corners = corner_peaks(response, exclude_border=False)
    assert len(corners) == 1

    corners = corner_peaks(response, exclude_border=False, min_distance=0)
    assert len(corners) == 4


def test_blank_image_nans():
    """Some of the corner detectors had a weakness in terms of returning
    NaN when presented with regions of constant intensity. This should
    be fixed by now. We test whether each detector returns something
    finite in the case of constant input"""

    detectors = [corner_moravec, corner_harris, corner_shi_tomasi,
                 corner_kitchen_rosenfeld, corner_foerstner]
    constant_image = np.zeros((20, 20))

    for det in detectors:
        response = det(constant_image)
        assert np.all(np.isfinite(response))


def test_corner_fast_image_unsupported_error():
    img = np.zeros((20, 20, 3))
    assert_raises(ValueError, corner_fast, img)


def test_corner_fast_lena():
    img = rgb2gray(data.lena())
    expected = np.array([[ 67, 157],
                         [204, 261],
                         [247, 146],
                         [269, 111],
                         [318, 158],
                         [386,  73],
                         [413,  70],
                         [435, 180],
                         [455, 177],
                         [461, 160]])
    actual = corner_peaks(corner_fast(img, 12, 0.3))
    assert_array_equal(actual, expected)


def test_corner_fast_orientation_image_unsupported_error():
    img = np.zeros((20, 20, 3))
    assert_raises(ValueError, corner_fast_orientation, img,
                  np.asarray([[7, 7]]))


def test_corner_fast_orientation_lena():
    img = rgb2gray(data.lena())
    corners = corner_peaks(corner_fast(img, 11, 0.35))
    expected = np.array([-1.9195897 , -3.03159624, -1.05991162, -2.89573739,
                         -2.61607644, 2.98660159])
    actual = corner_fast_orientation(img, corners)
    assert_almost_equal(actual, expected)


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
