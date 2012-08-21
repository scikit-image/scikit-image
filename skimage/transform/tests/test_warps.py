import sys
from numpy.testing import assert_array_almost_equal, run_module_suite
import numpy as np

from skimage.transform import (warp, fast_homography,
                               AffineTransform,
                               ProjectiveTransform,
                               SimilarityTransform)
from skimage import transform as tf, data, img_as_float
from skimage.color import rgb2gray


def test_warp():
    x = np.zeros((5, 5), dtype=np.uint8)
    x[2, 2] = 255
    x = img_as_float(x)
    theta = - np.pi / 2
    tform = SimilarityTransform(scale=1, rotation=theta, translation=(0, 4))

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

    x90 = warp(x,
            inverse_map=ProjectiveTransform(M).inverse,
            order=1)
    assert_array_almost_equal(x90, np.rot90(x))


def test_fast_homography():
    img = rgb2gray(data.lena())
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
        p0 = warp(img, ProjectiveTransform(H).inverse, mode=mode, order=1)
        p1 = fast_homography(img, H, mode=mode)

        # import matplotlib.pyplot as plt
        # f, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
        # ax0.imshow(img)
        # ax1.imshow(p0, cmap=plt.cm.gray)
        # ax2.imshow(p1, cmap=plt.cm.gray)
        # ax3.imshow(np.abs(p0 - p1), cmap=plt.cm.gray)
        # plt.show()

        d = np.mean(np.abs(p0 - p1))
        assert d < 0.001


def test_swirl():
    if not data.checkerboard().shape:
        print >> sys.stderr, ('Failed to read image data.checkerboard()'
                ' -- Skipping test_swirl')
        return
    image = img_as_float(data.checkerboard())

    swirl_params = {'radius': 80, 'rotation': 0, 'order': 2, 'mode': 'reflect'}
    swirled = tf.swirl(image, strength=10, **swirl_params)
    unswirled = tf.swirl(swirled, strength=-10, **swirl_params)

    assert np.mean(np.abs(image - unswirled)) < 0.01


def test_const_cval_out_of_range():
    img = np.random.randn(100, 100)
    warped = warp(img, AffineTransform(translation=(10, 10)), cval=-10)
    assert np.any(warped < 0)


def test_warp_identity():
    lena = data.lena()
    if not lena.shape:
        print >> sys.stderr, ('Failed to read image data.lena()'
                ' -- Skipping test_warp_identity')
        return
    lena = img_as_float(rgb2gray(lena))
    assert len(lena.shape) == 2
    assert np.allclose(lena,
            warp(lena, AffineTransform(rotation=0)))
    assert not np.allclose(lena,
            warp(lena, AffineTransform(rotation=0.1)))

    rgb_lena = np.transpose(
            np.asarray([lena, np.zeros_like(lena), lena]),
            (1, 2, 0))
    warped_rgb_lena = warp(rgb_lena, AffineTransform(rotation=0.1))

    assert np.allclose(rgb_lena,
            warp(rgb_lena, AffineTransform(rotation=0)))
    assert not np.allclose(rgb_lena, warped_rgb_lena)
    # assert no cross-talk between bands
    assert np.all(0 == warped_rgb_lena[:, :, 1])


if __name__ == "__main__":
    run_module_suite()
