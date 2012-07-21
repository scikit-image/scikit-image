from numpy.testing import assert_array_almost_equal, run_module_suite
import numpy as np

from skimage.transform import (warp, homography, fast_homography,
                               SimilarityTransform)
from skimage import transform as tf, data, img_as_float
from skimage.color import rgb2gray


def test_warp():
    x = np.zeros((5, 5), dtype=np.uint8)
    x[2, 2] = 255
    x = img_as_float(x)
    theta = -np.pi/2
    tform = SimilarityTransform(1, theta, (0, 4))

    x90 = warp(x, tform, order=1)
    assert_array_almost_equal(x90, np.rot90(x))

    x90 = warp(x, tform.inverse, order=1)
    assert_array_almost_equal(x90, np.rot90(x))


def test_homography():
    x = np.zeros((5, 5), dtype=np.uint8)
    x[1, 1] = 255
    x = img_as_float(x)
    theta = -np.pi/2
    M = np.array([[np.cos(theta),-np.sin(theta),0],
                  [np.sin(theta), np.cos(theta),4],
                  [0,             0,            1]])
    x90 = homography(x, M, order=1)
    assert_array_almost_equal(x90, np.rot90(x))


def test_fast_homography():
    img = rgb2gray(data.lena()).astype(np.uint8)
    img = img[:, :100]

    theta = np.deg2rad(30)
    scale = 0.5
    tx, ty = 50, 50

    H = np.eye(3)
    S = scale * np.sin(theta)
    C = scale * np.cos(theta)

    H[:2, :2] = [[C, -S], [S, C]]
    H[:2, 2] = [tx, ty]

    for mode in ('constant', 'mirror', 'wrap'):
        p0 = homography(img, H, mode=mode, order=1)
        p1 = fast_homography(img, H, mode=mode)
        p1 = np.round(p1)

        ## import matplotlib.pyplot as plt
        ## f, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
        ## ax0.imshow(img)
        ## ax1.imshow(p0, cmap=plt.cm.gray)
        ## ax2.imshow(p1, cmap=plt.cm.gray)
        ## ax3.imshow(np.abs(p0 - p1), cmap=plt.cm.gray)
        ## plt.show()

        d = np.mean(np.abs(p0 - p1))
        assert d < 0.2


def test_swirl():
    image = img_as_float(data.checkerboard())

    swirl_params = {'radius': 80, 'rotation': 0, 'order': 2, 'mode': 'reflect'}
    swirled = tf.swirl(image, strength=10, **swirl_params)
    unswirled = tf.swirl(swirled, strength=-10, **swirl_params)

    assert np.mean(np.abs(image - unswirled)) < 0.01


if __name__ == "__main__":
    run_module_suite()
