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


def test_selem_overflow():
    strel = np.ones((17, 17), dtype=np.uint8)
    img = np.zeros((20, 20))
    img[2:19, 2:19] = 1
    binary_res = binary.binary_erosion(img, strel)
    grey_res = img_as_bool(grey.erosion(img, strel))
    testing.assert_array_equal(binary_res, grey_res)


def test_out_argument():
    for func in (binary.binary_erosion, binary.binary_dilation):
        strel = np.ones((3, 3), dtype=np.uint8)
        img = np.ones((10, 10))
        out = np.zeros_like(img)
        out_saved = out.copy()
        func(img, strel, out=out)
        testing.assert_(np.any(out != out_saved))
        testing.assert_array_equal(out, func(img, strel))

if __name__ == '__main__':
    testing.run_module_suite()
