import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_equal
from scipy import ndimage as ndi

from skimage import data, color, morphology
from skimage._shared._warnings import expected_warnings
from skimage.util import img_as_bool


img = color.rgb2gray(data.astronaut())
bw_img = img > 100 / 255.


def test_non_square_image():
    footprint = morphology.square(3)
    binary_res = morphology.binary_erosion(bw_img[:100, :200], footprint)
    gray_res = img_as_bool(morphology.erosion(bw_img[:100, :200], footprint))
    assert_array_equal(binary_res, gray_res)


def test_binary_erosion():
    footprint = morphology.square(3)
    binary_res = morphology.binary_erosion(bw_img, footprint)
    gray_res = img_as_bool(morphology.erosion(bw_img, footprint))
    assert_array_equal(binary_res, gray_res)


def test_binary_dilation():
    footprint = morphology.square(3)
    binary_res = morphology.binary_dilation(bw_img, footprint)
    gray_res = img_as_bool(morphology.dilation(bw_img, footprint))
    assert_array_equal(binary_res, gray_res)


def test_binary_closing():
    footprint = morphology.square(3)
    binary_res = morphology.binary_closing(bw_img, footprint)
    gray_res = img_as_bool(morphology.closing(bw_img, footprint))
    assert_array_equal(binary_res, gray_res)


def test_binary_opening():
    footprint = morphology.square(3)
    binary_res = morphology.binary_opening(bw_img, footprint)
    gray_res = img_as_bool(morphology.opening(bw_img, footprint))
    assert_array_equal(binary_res, gray_res)


def test_footprint_overflow():
    footprint = np.ones((17, 17), dtype=np.uint8)
    img = np.zeros((20, 20), dtype=bool)
    img[2:19, 2:19] = True
    binary_res = morphology.binary_erosion(img, footprint)
    gray_res = img_as_bool(morphology.erosion(img, footprint))
    assert_array_equal(binary_res, gray_res)


def test_out_argument():
    for func in (morphology.binary_erosion, morphology.binary_dilation):
        footprint = np.ones((3, 3), dtype=np.uint8)
        img = np.ones((10, 10))
        out = np.zeros_like(img)
        out_saved = out.copy()
        func(img, footprint, out=out)
        assert np.any(out != out_saved)
        assert_array_equal(out, func(img, footprint))


binary_functions = [morphology.binary_erosion, morphology.binary_dilation,
                    morphology.binary_opening, morphology.binary_closing]


@pytest.mark.parametrize("function", binary_functions)
def test_default_footprint(function):
    footprint = morphology.diamond(radius=1)
    image = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], np.uint8)
    im_expected = function(image, footprint)
    im_test = function(image)
    assert_array_equal(im_expected, im_test)


def test_3d_fallback_default_footprint():
    # 3x3x3 cube inside a 7x7x7 image:
    image = np.zeros((7, 7, 7), bool)
    image[2:-2, 2:-2, 2:-2] = 1

    opened = morphology.binary_opening(image)

    # expect a "hyper-cross" centered in the 5x5x5:
    image_expected = np.zeros((7, 7, 7), dtype=bool)
    image_expected[2:5, 2:5, 2:5] = ndi.generate_binary_structure(3, 1)
    assert_array_equal(opened, image_expected)


binary_3d_fallback_functions = [morphology.binary_opening, morphology.binary_closing]


@pytest.mark.parametrize("function", binary_3d_fallback_functions)
def test_3d_fallback_cube_footprint(function):
    # 3x3x3 cube inside a 7x7x7 image:
    image = np.zeros((7, 7, 7), bool)
    image[2:-2, 2:-2, 2:-2] = 1

    cube = np.ones((3, 3, 3), dtype=np.uint8)

    new_image = function(image, cube)
    assert_array_equal(new_image, image)

def test_2d_ndimage_equivalence():
    image = np.zeros((9, 9), np.uint16)
    image[2:-2, 2:-2] = 2**14
    image[3:-3, 3:-3] = 2**15
    image[4, 4] = 2**16-1

    bin_opened = morphology.binary_opening(image)
    bin_closed = morphology.binary_closing(image)

    footprint = ndi.generate_binary_structure(2, 1)
    ndimage_opened = ndi.binary_opening(image, structure=footprint)
    ndimage_closed = ndi.binary_closing(image, structure=footprint)

    assert_array_equal(bin_opened, ndimage_opened)
    assert_array_equal(bin_closed, ndimage_closed)

def test_binary_output_2d():
    image = np.zeros((9, 9), np.uint16)
    image[2:-2, 2:-2] = 2**14
    image[3:-3, 3:-3] = 2**15
    image[4, 4] = 2**16-1

    bin_opened = morphology.binary_opening(image)
    bin_closed = morphology.binary_closing(image)

    int_opened = np.empty_like(image, dtype=np.uint8)
    int_closed = np.empty_like(image, dtype=np.uint8)
    morphology.binary_opening(image, out=int_opened)
    morphology.binary_closing(image, out=int_closed)

    assert_equal(bin_opened.dtype, bool)
    assert_equal(bin_closed.dtype, bool)

    assert_equal(int_opened.dtype, np.uint8)
    assert_equal(int_closed.dtype, np.uint8)

def test_binary_output_3d():
    image = np.zeros((9, 9, 9), np.uint16)
    image[2:-2, 2:-2, 2:-2] = 2**14
    image[3:-3, 3:-3, 3:-3] = 2**15
    image[4, 4, 4] = 2**16-1

    bin_opened = morphology.binary_opening(image)
    bin_closed = morphology.binary_closing(image)

    int_opened = np.empty_like(image, dtype=np.uint8)
    int_closed = np.empty_like(image, dtype=np.uint8)
    morphology.binary_opening(image, out=int_opened)
    morphology.binary_closing(image, out=int_closed)

    assert_equal(bin_opened.dtype, bool)
    assert_equal(bin_closed.dtype, bool)

    assert_equal(int_opened.dtype, np.uint8)
    assert_equal(int_closed.dtype, np.uint8)
