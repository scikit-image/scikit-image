from os.path import abspath, dirname, join as pjoin
import numpy as np
from scipy.signal import convolve2d
from skimage.data import camera
from skimage import deconvolution

test_img = camera().astype(np.float)


def test_wiener():
    psf = np.ones((5, 5))
    data = convolve2d(test_img, psf, 'same')
    np.random.seed(0)
    data += 0.1 * data.std() * np.random.standard_normal(data.shape)
    deconvolued = deconvolution.wiener(data, psf, 25)

    path = pjoin(dirname(abspath(__file__)), 'camera_wiener.npy')
    np.testing.assert_array_almost_equal(deconvolued, np.load(path))


def test_unsupervised_wiener():
    psf = np.ones((5, 5))
    data = convolve2d(test_img, psf, 'same')
    np.random.seed(0)
    data += 0.1 * data.std() * np.random.standard_normal(data.shape)
    deconvolued, _ = deconvolution.unsupervised_wiener(data, psf)

    path = pjoin(dirname(abspath(__file__)), 'camera_unsup.npy')
    np.testing.assert_array_almost_equal(deconvolued, np.load(path))


def test_richardson_lucy():
    return True
