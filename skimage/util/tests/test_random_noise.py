from numpy.testing import assert_array_equal, assert_allclose, assert_raises

import numpy as np
from skimage.data import camera
from skimage.util import random_noise, img_as_float


def test_set_seed():
    seed = 42
    cam = camera()
    test = random_noise(cam, seed=seed)
    assert_array_equal(test, random_noise(cam, seed=seed))


def test_salt():
    seed = 42
    cam = img_as_float(camera())
    cam_noisy = random_noise(cam, seed=seed, mode='salt', amount=0.15)
    saltmask = cam != cam_noisy

    # Ensure all changes are to 1.0
    assert_allclose(cam_noisy[saltmask], np.ones(saltmask.sum()))

    # Ensure approximately correct amount of noise was added
    proportion = float(saltmask.sum()) / (cam.shape[0] * cam.shape[1])
    assert 0.11 < proportion <= 0.18


def test_pepper():
    seed = 42
    cam = img_as_float(camera())
    cam_noisy = random_noise(cam, seed=seed, mode='pepper', amount=0.15)
    peppermask = cam != cam_noisy

    # Ensure all changes are to 1.0
    assert_allclose(cam_noisy[peppermask], np.zeros(peppermask.sum()))

    # Ensure approximately correct amount of noise was added
    proportion = float(peppermask.sum()) / (cam.shape[0] * cam.shape[1])
    assert 0.11 < proportion <= 0.18


def test_salt_and_pepper():
    seed = 42
    cam = img_as_float(camera())
    cam_noisy = random_noise(cam, seed=seed, mode='s&p', amount=0.15,
                             salt_vs_pepper=0.25)
    saltmask = np.logical_and(cam != cam_noisy, cam_noisy == 1.)
    peppermask = np.logical_and(cam != cam_noisy, cam_noisy == 0.)

    # Ensure all changes are to 0. or 1.
    assert_allclose(cam_noisy[saltmask], np.ones(saltmask.sum()))
    assert_allclose(cam_noisy[peppermask], np.zeros(peppermask.sum()))

    # Ensure approximately correct amount of noise was added
    proportion = float(
        saltmask.sum() + peppermask.sum()) / (cam.shape[0] * cam.shape[1])
    assert 0.11 < proportion <= 0.18

    # Verify the relative amount of salt vs. pepper is close to expected
    assert 0.18 < saltmask.sum() / float(peppermask.sum()) < 0.32


def test_gaussian():
    seed = 42
    data = np.zeros((128, 128)) + 0.5
    data_gaussian = random_noise(data, seed=seed, var=0.01)
    assert 0.008 < data_gaussian.var() < 0.012

    data_gaussian = random_noise(data, seed=seed, mean=0.3, var=0.015)
    assert 0.28 < data_gaussian.mean() - 0.5 < 0.32
    assert 0.012 < data_gaussian.var() < 0.018


def test_speckle():
    seed = 42
    data = np.zeros((128, 128)) + 0.1
    np.random.seed(seed=42)
    noise = np.random.normal(0.1, 0.02 ** 0.5, (128, 128))
    expected = np.clip(data + data * noise, 0, 1)

    data_speckle = random_noise(data, mode='speckle', seed=seed, mean=0.1,
                                var=0.02)
    assert_allclose(expected, data_speckle)


def test_bad_mode():
    data = np.zeros((64, 64))
    assert_raises(KeyError, random_noise, data, 'perlin')


if __name__ == '__main__':
    np.testing.run_module_suite()
