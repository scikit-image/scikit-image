import numpy as np
from numpy import testing

from skimage import data, color
from skimage.util import img_as_bool
from skimage.morphology import binary, grey, selem


lena = color.rgb2gray(data.lena())
bw_lena = lena > 100


def test_non_square_image():
    strel = selem.square(3)
    binary_res = binary.binary_erosion(bw_lena[:100, :200], strel)
    grey_res = img_as_bool(grey.erosion(bw_lena[:100, :200], strel))
    testing.assert_array_equal(binary_res, grey_res)


def test_binary_erosion():
    strel = selem.square(3)
    binary_res = binary.binary_erosion(bw_lena, strel)
    grey_res = img_as_bool(grey.erosion(bw_lena, strel))
    testing.assert_array_equal(binary_res, grey_res)


def test_binary_dilation():
    strel = selem.square(3)
    binary_res = binary.binary_dilation(bw_lena, strel)
    grey_res = img_as_bool(grey.dilation(bw_lena, strel))
    testing.assert_array_equal(binary_res, grey_res)


def test_binary_closing():
    strel = selem.square(3)
    binary_res = binary.binary_closing(bw_lena, strel)
    grey_res = img_as_bool(grey.closing(bw_lena, strel))
    testing.assert_array_equal(binary_res, grey_res)


def test_binary_opening():
    strel = selem.square(3)
    binary_res = binary.binary_opening(bw_lena, strel)
    grey_res = img_as_bool(grey.opening(bw_lena, strel))
    testing.assert_array_equal(binary_res, grey_res)


if __name__ == '__main__':
    testing.run_module_suite()
