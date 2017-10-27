from numpy.testing import assert_equal, assert_almost_equal
import pytest
import numpy as np

from skimage._shared._warnings import expected_warnings
from skimage import draw
from skimage.measure import (moments, moments_central, moments_normalized,
                             moments_hu)


def test_moments():
    image = np.zeros((20, 20), dtype=np.double)
    image[14, 14] = 1
    image[15, 15] = 1
    image[14, 15] = 0.5
    image[15, 14] = 0.5
    m = moments(image)
    assert_equal(m[0, 0], 3)
    assert_almost_equal(m[1, 0] / m[0, 0], 14.5)
    assert_almost_equal(m[0, 1] / m[0, 0], 14.5)


def test_moments_central():
    image = np.zeros((20, 20), dtype=np.double)
    image[14, 14] = 1
    image[15, 15] = 1
    image[14, 15] = 0.5
    image[15, 14] = 0.5
    mu = moments_central(image, (14.5, 14.5))

    # shift image by dx=2, dy=2
    image2 = np.zeros((20, 20), dtype=np.double)
    image2[16, 16] = 1
    image2[17, 17] = 1
    image2[16, 17] = 0.5
    image2[17, 16] = 0.5
    mu2 = moments_central(image2, (14.5 + 2, 14.5 + 2))
    # central moments must be translation invariant
    assert_equal(mu, mu2)


def test_moments_central_deprecated():
    image = np.zeros((20, 20), dtype=np.double)
    image[5:-5, 5:-5] = np.random.random((10, 10))
    center = moments(image, 1)[[1, 0], [0, 1]]
    cr, cc = center
    with expected_warnings(['deprecated 2D-only']):
        mu0 = moments_central(image, cr, cc)
        mu1 = moments_central(image, cr=cr, cc=cc)
    mu_ref = moments_central(image, center)
    assert_almost_equal(mu0.T, mu_ref)
    assert_almost_equal(mu1.T, mu_ref)


def test_moments_contour():
    image = np.zeros((20, 20), dtype=np.double)
    image[13:17, 13:17] = 1
    mu_image = moments(image)

    contour = np.array([[r, c] for r in range(13, 17)
                        for c in range(13, 17)], dtype=np.double)
    mu_contour = moments_contour(contour)
    assert_almost_equal(mu_contour, mu_image)


def test_moments_central_contour():
    image = np.zeros((20, 20), dtype=np.double)
    image[13:17, 13:17] = 1
    mu_image = moments_central(image, (3, 3))

    contour = np.array([[r, c] for r in range(13, 17)
                        for c in range(13, 17)], dtype=np.double)
    mu_contour = moments_contour_central(contour, (3, 3))
    assert_almost_equal(mu_contour, mu_image)

    # shift image by dx=3 dy=3
    image = np.zeros((20, 20), dtype=np.double)
    image[16:20, 16:20] = 1
    mu_image = moments_central(image, (3, 3))

    contour = np.array([[r, c] for r in range(16, 20)
                        for c in range(16, 20)], dtype=np.double)
    mu_contour = moments_contour_central(contour, (3, 3))
    assert_almost_equal(mu_contour, mu_image)


def test_moments_normalized():
    image = np.zeros((20, 20), dtype=np.double)
    image[13:17, 13:17] = 1
    mu = moments_central(image, (14.5, 14.5))
    nu = moments_normalized(mu)
    # shift image by dx=-3, dy=-3 and scale by 0.5
    image2 = np.zeros((20, 20), dtype=np.double)
    image2[11:13, 11:13] = 1
    mu2 = moments_central(image2, (11.5, 11.5))
    nu2 = moments_normalized(mu2)
    # central moments must be translation and scale invariant
    assert_almost_equal(nu, nu2, decimal=1)


def test_moments_normalized_3d():
    image = draw.ellipsoid(1, 1, 10)
    mu = moments_central(image)
    nu = moments_normalized(mu)
    assert nu[0, 0, 2] > nu[0, 2, 0]
    assert_almost_equal(nu[0, 2, 0], nu[2, 0, 0])


def test_moments_normalized_invalid():
    with pytest.raises(ValueError):
        moments_normalized(np.zeros((3, 3)), 3)
    with pytest.raises(ValueError):
        moments_normalized(np.zeros((3, 3)), 4)


def test_moments_hu():
    image = np.zeros((20, 20), dtype=np.double)
    image[13:15, 13:17] = 1
    mu = moments_central(image, (13.5, 14.5))
    nu = moments_normalized(mu)
    hu = moments_hu(nu)
    # shift image by dx=2, dy=3, scale by 0.5 and rotate by 90deg
    image2 = np.zeros((20, 20), dtype=np.double)
    image2[11, 11:13] = 1
    image2 = image2.T
    mu2 = moments_central(image2, (11.5, 11))
    nu2 = moments_normalized(mu2)
    hu2 = moments_hu(nu2)
    # central moments must be translation and scale invariant
    assert_almost_equal(hu, hu2, decimal=1)


if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
