import warnings

import numpy as np
import numpy.testing.assert_array_almost_equal

from scipy.signal import convolve2d as conv2
from skimage import data, deconvolution

# Test deconvolution
# ===========================

test_img = data.camera().astype(np.float)


def test_wiener():
    psf = np.ones((5, 5))
    data = conv2(test_img, psf, 'same')
    np.random.seed(0)
    data += 0.1 * data.std() * np.random.standard_normal(data.shape)
    deconvolued = deconvolution.wiener(data, psf, 25)

    numpy.testing.assert_array_almost_equal(deconvolued,
                                            np.load("./camera_wiener.npy"))


def test_unsupervised_wiener():
    psf = np.ones((5, 5))
    data = conv2(test_img, psf, 'same')
    np.random.seed(0)
    data += 0.1 * data.std() * np.random.standard_normal(data.shape)
    deconvolued, _ = deconvolution.unsupervised_wiener(data, psf)

    numpy.testing.assert_array_almost_equal(deconvolued,
                                            np.load("./camera_unsup.npy"))


def test_rychardson_lucy():
    return True
