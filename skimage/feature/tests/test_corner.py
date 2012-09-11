import numpy as np

from skimage import data
from skimage import img_as_float

from skimage.feature import (corner_moravec, corner_harris, corner_shi_tomasi,
                             peak_local_max)


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


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
