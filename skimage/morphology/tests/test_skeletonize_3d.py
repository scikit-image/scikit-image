from __future__ import division, print_function, absolute_import

import os
import warnings

import numpy as np
from numpy.testing import (assert_equal, run_module_suite, assert_raises,
                           assert_)

import scipy.ndimage as ndi

import skimage
from skimage import io, draw, data_dir
#from skimage import draw
from skimage.util import img_as_ubyte

from skimage.morphology import skeletonize_3d


# basic behavior tests (mostly copied over from 2D skeletonize)

def test_skeletonize_wrong_dim():
    im = np.zeros(5, dtype=np.uint8)
    assert_raises(ValueError, skeletonize_3d, im)

    im = np.zeros((5, 5, 5, 5), dtype=np.uint8)
    assert_raises(ValueError, skeletonize_3d, im)


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


# nose test generators:
# 2D images
def test_simple_2d_images():
    for fname in ("strip", "loop", "cross", "two-hole"):
        yield check_skel, fname


# trivial 3D images
def test_simple_3d():
    for fname in ['3/stack', '4/stack']:
        yield check_skel_3d, fname


# 'slow' test: Bat Cochlea from FIJI collections.
def test_large():
    for fname in ['bat/bat-cochlea-volume']:
        yield check_skel_3d, fname


def get_data_path():
    # XXX this is a bad temp hack
    return os.path.join(os.path.split(skimage.__file__)[0],
                        'morphology',
                        'tests',
                        'data')


def check_skel(fname):
    # compute the thin image and compare the result to that of ImageJ
    img = np.loadtxt(os.path.join(get_data_path(), fname + '.txt'),
                     dtype=np.uint8)

    # compute
    img1_2d = skeletonize_3d(img)

    # and compare to FIJI
    img_f = np.loadtxt(os.path.join(get_data_path(), fname + '_fiji.txt'),
                       dtype=np.uint8)

    assert_equal(img1_2d, img_f)


def check_skel_3d(fname):
    img = io.imread(os.path.join(get_data_path(), fname + '.tif'))
    img_f = io.imread(os.path.join(get_data_path(), fname + '_fiji.tif'))

    img_s = skeletonize_3d(img)
    assert_equal(img_s, img_f)


if __name__ == '__main__':
    run_module_suite()
