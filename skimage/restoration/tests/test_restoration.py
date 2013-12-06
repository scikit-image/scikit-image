from os.path import abspath, dirname, join as pjoin

import numpy as np
from scipy.signal import convolve2d

import skimage
from skimage.data import camera
from skimage import restoration
from skimage.restoration import uft

test_img = skimage.img_as_float(camera())


def test_wiener():
    psf = np.ones((5, 5)) / 25
    data = convolve2d(test_img, psf, 'same')
    np.random.seed(0)
    data += 0.1 * data.std() * np.random.standard_normal(data.shape)
    deconvolved = restoration.wiener(data, psf, 0.05)

    path = pjoin(dirname(abspath(__file__)), 'camera_wiener.npy')
    np.testing.assert_allclose(deconvolved, np.load(path), rtol=1e-3)

    _, laplacian = uft.laplacian(2, data.shape)
    otf = uft.ir2tf(psf, data.shape, is_real=False)
    deconvolved = restoration.wiener(data, otf, 0.05,
                                     reg=laplacian,
                                     is_real=False)
    np.testing.assert_allclose(np.real(deconvolved),
                               np.load(path),
                               rtol=1e-3)


def test_unsupervised_wiener():
    psf = np.ones((5, 5)) / 25
    data = convolve2d(test_img, psf, 'same')
    np.random.seed(0)
    data += 0.1 * data.std() * np.random.standard_normal(data.shape)
    deconvolved, _ = restoration.unsupervised_wiener(data, psf)

    path = pjoin(dirname(abspath(__file__)), 'camera_unsup.npy')
    np.testing.assert_allclose(deconvolved, np.load(path), rtol=1e-3)

    _, laplacian = uft.laplacian(2, data.shape)
    otf = uft.ir2tf(psf, data.shape, is_real=False)
    np.random.seed(0)
    deconvolved = restoration.unsupervised_wiener(
        data, otf, reg=laplacian, is_real=False,
        user_params={"callback": lambda x: None})[0]
    path = pjoin(dirname(abspath(__file__)), 'camera_unsup2.npy')
    np.testing.assert_allclose(np.real(deconvolved),
                               np.load(path),
                               rtol=1e-3)


def test_richardson_lucy():
    psf = np.ones((5, 5)) / 25
    data = convolve2d(test_img, psf, 'same')
    np.random.seed(0)
    data += 0.1 * data.std() * np.random.standard_normal(data.shape)
    deconvolved = restoration.richardson_lucy(data, psf, 5)

    path = pjoin(dirname(abspath(__file__)), 'camera_rl.npy')
    np.testing.assert_allclose(deconvolved, np.load(path), rtol=1e-3)
