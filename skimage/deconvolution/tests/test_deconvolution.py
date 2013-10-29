from os.path import abspath, dirname, join as pjoin

import numpy as np
from scipy.signal import convolve2d

import skimage
from skimage.data import camera
from skimage import deconvolution

test_img = skimage.img_as_float(camera())


def test_wiener():
    psf = np.ones((5, 5)) / 25
    data = convolve2d(test_img, psf, 'same')
    np.random.seed(0)
    data += 0.1 * data.std() * np.random.standard_normal(data.shape)
    deconvolved = deconvolution.wiener(data, psf, 0.05)

    path = pjoin(dirname(abspath(__file__)), 'camera_wiener.npy')
    np.testing.assert_allclose(deconvolved, np.load(path))


def test_unsupervised_wiener():
    psf = np.ones((5, 5)) / 25
    data = convolve2d(test_img, psf, 'same')
    np.random.seed(0)
    data += 0.1 * data.std() * np.random.standard_normal(data.shape)
    deconvolved, _ = deconvolution.unsupervised_wiener(data, psf)

    path = pjoin(dirname(abspath(__file__)), 'camera_unsup.npy')
    np.testing.assert_allclose(deconvolved, np.load(path))


def test_richardson_lucy():
    psf = np.ones((5, 5)) / 25
    data = convolve2d(test_img, psf, 'same')
    np.random.seed(0)
    data += 0.1 * data.std() * np.random.standard_normal(data.shape)
    deconvolved = deconvolution.richardson_lucy(data, psf, 5)

    path = pjoin(dirname(abspath(__file__)), 'camera_rl.npy')
    np.testing.assert_allclose(deconvolved, np.load(path))
