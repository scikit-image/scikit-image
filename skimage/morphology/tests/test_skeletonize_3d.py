from __future__ import division, print_function, absolute_import

import os
import warnings

import numpy as np
from numpy.testing import (assert_equal, run_module_suite, assert_raises,
                           assert_)

import scipy.ndimage as ndi

import skimage
from skimage import io, draw, data_dir
from skimage.data import binary_blobs
from skimage.util import img_as_ubyte

from skimage.morphology import skeletonize_3d


# basic behavior tests (mostly copied over from 2D skeletonize)

def test_skeletonize_wrong_dim():
    im = np.zeros(5, dtype=np.uint8)
    assert_raises(ValueError, skeletonize_3d, im)

    im = np.zeros((5, 5, 5, 5), dtype=np.uint8)
    assert_raises(ValueError, skeletonize_3d, im)


def test_skeletonize_1D():
    # a corner case of an image of a shape(1, N)
    im = np.ones((5, 1), dtype=np.uint8)
    res = skeletonize_3d(im)
    assert_equal(res, im)


def test_skeletonize_no_foreground():
    im = np.zeros((5, 5), dtype=np.uint8)
    result = skeletonize_3d(im)
    assert_equal(result, im)


def test_skeletonize_all_foreground():
    im = np.ones((3, 4), dtype=np.uint8)
    assert_equal(skeletonize_3d(im),
                 np.array([[0, 0, 0, 0],
                           [1, 1, 1, 1],
                           [0, 0, 0, 0]], dtype=np.uint8))


def test_skeletonize_single_point():
    im = np.zeros((5, 5), dtype=np.uint8)
    im[3, 3] = 1
    result = skeletonize_3d(im)
    assert_equal(result, im)


def test_skeletonize_already_thinned():
    im = np.zeros((5, 5), dtype=np.uint8)
    im[3, 1:-1] = 1
    im[2, -1] = 1
    im[4, 0] = 1
    result = skeletonize_3d(im)
    assert_equal(result, im)


def test_dtype_conv():
    # check that the operation does the right thing with floats etc
    # also check non-contiguous input
    img = np.random.random((16, 16))[::2, ::2]
    img[img < 0.5] = 0

    orig = img.copy()

    with warnings.catch_warnings():
        # UserWarning for possible precision loss, expected
        warnings.simplefilter('ignore', UserWarning)
        res = skeletonize_3d(img)

    assert_equal(res.dtype, np.uint8)
    assert_equal(img, orig)  # operation does not clobber the original 
    assert_equal(res.max(),
                 img_as_ubyte(img).max())    # the intensity range is preserved


def test_input():
    # check that the input is not clobbered
    # for 2D and 3D images of varying dtypes
    imgs = [np.ones((8, 8), dtype=float), np.ones((4, 8, 8), dtype=float),
            np.ones((8, 8), dtype=np.uint8), np.ones((4, 8, 8), dtype=np.uint8),
            np.ones((8, 8), dtype=bool), np.ones((4, 8, 8), dtype=bool)]
    for img in imgs:
        yield check_input, img


def check_input(img):
    orig = img.copy()
    with warnings.catch_warnings():
        # UserWarning for possible precision loss, expected
        warnings.simplefilter('ignore', UserWarning)
        res = skeletonize_3d(img)
    assert_equal(img, orig)


def test_skeletonize_num_neighbours():
    # an empty image
    image = np.zeros((300, 300))

    # foreground object 1
    image[10:-10, 10:100] = 1
    image[-100:-10, 10:-10] = 1
    image[10:-10, -100:-10] = 1

    # foreground object 2
    rs, cs = draw.line(250, 150, 10, 280)
    for i in range(10):
        image[rs + i, cs] = 1
    rs, cs = draw.line(10, 150, 250, 280)
    for i in range(20):
        image[rs + i, cs] = 1

    # foreground object 3
    ir, ic = np.indices(image.shape)
    circle1 = (ic - 135)**2 + (ir - 150)**2 < 30**2
    circle2 = (ic - 135)**2 + (ir - 150)**2 < 20**2
    image[circle1] = 1
    image[circle2] = 0
    result = skeletonize_3d(image)

    # there should never be a 2x2 block of foreground pixels in a skeleton
    mask = np.array([[1,  1],
                     [1,  1]], np.uint8)
    blocks = ndi.correlate(result, mask, mode='constant')
    assert_(not np.any(blocks == 4))


def test_two_hole_image():
    # test a simple 2D image against FIJI
    img_o = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                      [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                      [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                      [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                      [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
                      [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                      dtype=np.uint8)
    img_f = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                      dtype=np.uint8)
    res = skeletonize_3d(img_o)
    assert_equal(res, img_f)


def test_3d_vs_fiji():
    # generate an image with blobs and compate its skeleton to
    # the skeleton generated by FIJI
    img = binary_blobs(32, 0.05, n_dim=3, seed=1234)
    img = img[:-2, ...]
    img = img.astype(np.uint8)*255

    img_s = skeletonize_3d(img)
    img_f = io.imread(os.path.join(data_dir, "_blobs_3d_fiji_skeleton.tif"))
    assert_equal(img_s, img_f)


if __name__ == '__main__':
    run_module_suite()
