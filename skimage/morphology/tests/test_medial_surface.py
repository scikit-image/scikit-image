import numpy as np
import scipy.ndimage as ndi

from skimage import io, draw
from skimage.data import binary_blobs
from skimage.util import img_as_ubyte
from skimage.morphology import medial_surface

from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_, parametrize, fetch

# basic behavior tests (mostly copied over from test_skeletonize_3d)


def test_medial_surface_wrong_dim():
    im = np.zeros(5, dtype=np.uint8)
    with testing.raises(ValueError):
        medial_surface(im)

    im = np.zeros((5, 5), dtype=np.uint8)
    with testing.raises(ValueError):
        medial_surface(im)

    im = np.zeros((5, 5, 5, 5), dtype=np.uint8)
    with testing.raises(ValueError):
        medial_surface(im)


def test_medial_surface_1D():
    # a corner case of an image of a shape(1, 1, N)
    im = np.ones((5, 1, 1), dtype=np.uint8)
    res = medial_surface(im)
    assert_equal(res, im)


def test_medial_surface_no_foreground():
    im = np.zeros((5, 5, 5), dtype=np.uint8)
    result = medial_surface(im)
    assert_equal(result, im)


def test_medial_surface_all_foreground():
    im = np.ones((3, 3, 3), dtype=np.uint8)
    assert_equal(medial_surface(im),
                 np.array([[[0, 0, 0],
                            [1, 1, 1],
                            [0, 0, 0]],
                           [[0, 0, 0],
                            [1, 1, 1],
                            [0, 0, 0]],
                           [[0, 0, 0],
                            [1, 1, 1],
                            [0, 0, 0]]], dtype=np.uint8))


def test_medial_surface_single_point():
    im = np.zeros((5, 5, 5), dtype=np.uint8)
    im[3, 3, 3] = 1
    result = medial_surface(im)
    assert_equal(result, im)


def test_medial_surface_already_thinned():
    im = np.zeros((5, 5, 5), dtype=np.uint8)
    im[1:-1, 1:-1, 3] = 1
    result = medial_surface(im)
    assert_equal(result, im)

    im[1:-1, 1:-1, 3] = 0
    im[1:-1, 3, 1:-1] = 1
    result = medial_surface(im)
    assert_equal(result, im)

    im[1:-1, 3, 1:-1] = 0
    im[3, 1:-1, 1:-1] = 1
    result = medial_surface(im)
    assert_equal(result, im)


def test_dtype_conv():
    # check that the operation does the right thing with floats etc
    # also check non-contiguous input
    img = np.random.random((16, 16, 16))[::2, ::2, ::2]
    img[img < 0.5] = 0

    orig = img.copy()
    res = medial_surface(img)
    img_max = img_as_ubyte(img).max()

    assert_equal(res.dtype, np.uint8)
    assert_equal(img, orig)  # operation does not clobber the original
    assert_equal(res.max(), img_max)    # the intensity range is preserved


@parametrize("img", [
    np.ones((4, 8, 8), dtype=float)
])
def test_input_with_warning(img):
    # check that the input is not clobbered
    # for 3D images of varying dtypes
    check_input(img)


@parametrize("img", [
    np.ones((4, 8, 8), dtype=np.uint8),
    np.ones((4, 8, 8), dtype=bool)
])
def test_input_without_warning(img):
    # check that the input is not clobbered
    # for 2D and 3D images of varying dtypes
    check_input(img)


def check_input(img):
    orig = img.copy()
    medial_surface(img)
    assert_equal(img, orig)
