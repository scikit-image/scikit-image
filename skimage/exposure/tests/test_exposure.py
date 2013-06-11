import warnings

import numpy as np
from numpy.testing import assert_array_almost_equal as assert_close
from numpy.testing import assert_array_equal
import skimage
from skimage import data
from skimage import exposure
from skimage.color import rgb2gray
from skimage.util.dtype import dtype_range


# Test histogram equalization
# ===========================

# squeeze image intensities to lower image contrast
test_img = skimage.img_as_float(data.camera())
test_img = exposure.rescale_intensity(test_img / 5. + 100)


def test_equalize_ubyte():
    img = skimage.img_as_ubyte(test_img)
    img_eq = exposure.equalize_hist(img)

    cdf, bin_edges = exposure.cumulative_distribution(img_eq)
    check_cdf_slope(cdf)


def test_equalize_float():
    img = skimage.img_as_float(test_img)
    img_eq = exposure.equalize_hist(img)

    cdf, bin_edges = exposure.cumulative_distribution(img_eq)
    check_cdf_slope(cdf)


def check_cdf_slope(cdf):
    """Slope of cdf which should equal 1 for an equalized histogram."""
    norm_intensity = np.linspace(0, 1, len(cdf))
    slope, intercept = np.polyfit(norm_intensity, cdf, 1)
    assert 0.9 < slope < 1.1


# Test rescale intensity
# ======================

def test_rescale_stretch():
    image = np.array([51, 102, 153], dtype=np.uint8)
    out = exposure.rescale_intensity(image)
    assert out.dtype == np.uint8
    assert_close(out, [0, 127, 255])


def test_rescale_shrink():
    image = np.array([51., 102., 153.])
    out = exposure.rescale_intensity(image)
    assert_close(out, [0, 0.5, 1])


def test_rescale_in_range():
    image = np.array([51., 102., 153.])
    out = exposure.rescale_intensity(image, in_range=(0, 255))
    assert_close(out, [0.2, 0.4, 0.6])


def test_rescale_in_range_clip():
    image = np.array([51., 102., 153.])
    out = exposure.rescale_intensity(image, in_range=(0, 102))
    assert_close(out, [0.5, 1, 1])


def test_rescale_out_range():
    image = np.array([-10, 0, 10], dtype=np.int8)
    out = exposure.rescale_intensity(image, out_range=(0, 127))
    assert out.dtype == np.int8
    assert_close(out, [0, 63, 127])


# Test adaptive histogram equalization
# ====================================

def test_adapthist_scalar():
    '''Test a scalar uint8 image
    '''
    img = skimage.img_as_ubyte(data.moon())
    adapted = exposure.equalize_adapthist(img, clip_limit=0.02)
    assert adapted.min() == 0
    assert adapted.max() == (1 << 16) - 1
    assert img.shape == adapted.shape
    full_scale = skimage.exposure.rescale_intensity(skimage.img_as_uint(img))

    assert_almost_equal = np.testing.assert_almost_equal
    assert_almost_equal(peak_snr(full_scale, adapted), 101.231, 3)
    assert_almost_equal(norm_brightness_err(full_scale, adapted),
                        0.041, 3)
    return img, adapted


def test_adapthist_grayscale():
    '''Test a grayscale float image
    '''
    img = skimage.img_as_float(data.lena())
    img = rgb2gray(img)
    img = np.dstack((img, img, img))
    adapted = exposure.equalize_adapthist(img, 10, 9, clip_limit=0.01,
                        nbins=128)
    assert_almost_equal = np.testing.assert_almost_equal
    assert img.shape == adapted.shape
    assert_almost_equal(peak_snr(img, adapted), 97.531, 3)
    assert_almost_equal(norm_brightness_err(img, adapted), 0.0313, 3)
    return data, adapted


def test_adapthist_color():
    '''Test an RGB color uint16 image
    '''
    img = skimage.img_as_uint(data.lena())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        hist, bin_centers = exposure.histogram(img)
        assert len(w) > 0
    adapted = exposure.equalize_adapthist(img, clip_limit=0.01)
    assert_almost_equal = np.testing.assert_almost_equal
    assert adapted.min() == 0
    assert adapted.max() == 1.0
    assert img.shape == adapted.shape
    full_scale = skimage.exposure.rescale_intensity(img)
    assert_almost_equal(peak_snr(full_scale, adapted), 102.940, 3)
    assert_almost_equal(norm_brightness_err(full_scale, adapted),
                        0.0110, 3)
    return data, adapted


def peak_snr(img1, img2):
    '''Peak signal to noise ratio of two images

    Parameters
    ----------
    img1 : array-like
    img2 : array-like

    Returns
    -------
    peak_snr : float
        Peak signal to noise ratio
    '''
    if img1.ndim == 3:
        img1, img2 = rgb2gray(img1.copy()), rgb2gray(img2.copy())
    img1 = skimage.img_as_float(img1)
    img2 = skimage.img_as_float(img2)
    mse = 1. / img1.size * np.square(img1 - img2).sum()
    _, max_ = dtype_range[img1.dtype.type]
    return 20 * np.log(max_ / mse)


def norm_brightness_err(img1, img2):
    '''Normalized Absolute Mean Brightness Error between two images

    Parameters
    ----------
    img1 : array-like
    img2 : array-like

    Returns
    -------
    norm_brightness_error : float
        Normalized absolute mean brightness error
    '''
    if img1.ndim == 3:
        img1, img2 = rgb2gray(img1), rgb2gray(img2)
    ambe = np.abs(img1.mean() - img2.mean())
    nbe = ambe / dtype_range[img1.dtype.type][1]
    return nbe


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()


# Test Gamma Correction
# =====================

def test_adjust_gamma_one():
    """Same image should be returned for gamma equal to one"""
    image = np.random.uniform(0, 255, (8, 8))
    result = exposure.adjust_gamma(image, 1)
    assert_array_equal(result, image)


def test_adjust_gamma_zero():
    """White image should be returned for gamma equal to zero"""
    image = np.random.uniform(0, 255, (8, 8))
    result = exposure.adjust_gamma(image, 0)
    dtype = image.dtype.type
    assert_array_equal(result, dtype_range[dtype][1])


def test_adjust_gamma_less_one():
    """Verifying the output with expected results for gamma
    correction with gamma equal to half"""
    image = np.arange(0, 255, 4, np.uint8).reshape(8,8)
    expected = np.array([[  0,  31,  45,  55,  63,  71,  78,  84],
        [ 90,  95, 100, 105, 110, 115, 119, 123],
        [127, 131, 135, 139, 142, 146, 149, 153],
        [156, 159, 162, 165, 168, 171, 174, 177],
        [180, 183, 186, 188, 191, 194, 196, 199],
        [201, 204, 206, 209, 211, 214, 216, 218],
        [221, 223, 225, 228, 230, 232, 234, 236],
        [238, 241, 243, 245, 247, 249, 251, 253]], dtype=np.uint8)

    result = exposure.adjust_gamma(image, 0.5)
    assert_array_equal(result, expected)


def test_adjust_gamma_greater_one():
    """Verifying the output with expected results for gamma
    correction with gamma equal to two"""
    image = np.arange(0, 255, 4, np.uint8).reshape(8,8)
    expected = np.array([[  0,   0,   0,   0,   1,   1,   2,   3],
        [  4,   5,   6,   7,   9,  10,  12,  14],
        [ 16,  18,  20,  22,  25,  27,  30,  33],
        [ 36,  39,  42,  45,  49,  52,  56,  60],
        [ 64,  68,  72,  76,  81,  85,  90,  95],
        [100, 105, 110, 116, 121, 127, 132, 138],
        [144, 150, 156, 163, 169, 176, 182, 189],
        [196, 203, 211, 218, 225, 233, 241, 249]], dtype=np.uint8)

    result = exposure.adjust_gamma(image, 2)
    assert_array_equal(result, expected)


# Test Logarithmic Correction
# ===========================

def test_adjust_log():
    """Verifying the output with expected results for logarithmic
    correction with multiplier constant multiplier equal to unity"""
    image = np.arange(0, 255, 4, np.uint8).reshape(8,8)
    expected = np.array([[  0,   5,  11,  16,  22,  27,  33,  38],
        [ 43,  48,  53,  58,  63,  68,  73,  77],
        [ 82,  86,  91,  95, 100, 104, 109, 113],
        [117, 121, 125, 129, 133, 137, 141, 145],
        [149, 153, 157, 160, 164, 168, 172, 175],
        [179, 182, 186, 189, 193, 196, 199, 203],
        [206, 209, 213, 216, 219, 222, 225, 228],
        [231, 234, 238, 241, 244, 246, 249, 252]], dtype=np.uint8)

    result = exposure.adjust_log(image, 1)
    assert_array_equal(result, expected)


def test_adjust_inv_log():
    """Verifying the output with expected results for inverse logarithmic
    correction with multiplier constant multiplier equal to unity"""
    image = np.arange(0, 255, 4, np.uint8).reshape(8,8)
    expected = np.array([[  0,   2,   5,   8,  11,  14,  17,  20],
        [ 23,  26,  29,  32,  35,  38,  41,  45],
        [ 48,  51,  55,  58,  61,  65,  68,  72],
        [ 76,  79,  83,  87,  90,  94,  98, 102],
        [106, 110, 114, 118, 122, 126, 130, 134],
        [138, 143, 147, 151, 156, 160, 165, 170],
        [174, 179, 184, 188, 193, 198, 203, 208],
        [213, 218, 224, 229, 234, 239, 245, 250]], dtype=np.uint8)

    result = exposure.adjust_log(image, 1, True)
    assert_array_equal(result, expected)


# Test Sigmoid Correction
# =======================

def test_adjust_sigmoid_cutoff_one():
    """Verifying the output with expected results for sigmoid correction
    with cutoff equal to one and gain of 5"""
    image = np.arange(0, 255, 4, np.uint8).reshape(8,8)
    expected = np.array([[  1,   1,   1,   2,   2,   2,   2,   2],
        [  3,   3,   3,   4,   4,   4,   5,   5],
        [  5,   6,   6,   7,   7,   8,   9,  10],
        [ 10,  11,  12,  13,  14,  15,  16,  18],
        [ 19,  20,  22,  24,  25,  27,  29,  32],
        [ 34,  36,  39,  41,  44,  47,  50,  54],
        [ 57,  61,  64,  68,  72,  76,  80,  85],
        [ 89,  94,  99, 104, 108, 113, 118, 123]], dtype=np.uint8)

    result = exposure.adjust_sigmoid(image, 1, 5)
    assert_array_equal(result, expected)


def test_adjust_sigmoid_cutoff_zero():
    """Verifying the output with expected results for sigmoid correction
    with cutoff equal to zero and gain of 10"""
    image = np.arange(0, 255, 4, np.uint8).reshape(8,8)
    expected = np.array([[127, 137, 147, 156, 166, 175, 183, 191],
        [198, 205, 211, 216, 221, 225, 229, 232],
        [235, 238, 240, 242, 244, 245, 247, 248],
        [249, 250, 250, 251, 251, 252, 252, 253],
        [253, 253, 253, 253, 254, 254, 254, 254],
        [254, 254, 254, 254, 254, 254, 254, 254],
        [254, 254, 254, 254, 254, 254, 254, 254],
        [254, 254, 254, 254, 254, 254, 254, 254]], dtype=np.uint8)

    result = exposure.adjust_sigmoid(image, 0, 10)
    assert_array_equal(result, expected)


def test_adjust_sigmoid_cutoff_half():
    """Verifying the output with expected results for sigmoid correction
    with cutoff equal to half and gain of 10"""
    image = np.arange(0, 255, 4, np.uint8).reshape(8,8)
    expected = np.array([[  1,   1,   2,   2,   3,   3,   4,   5],
        [  5,   6,   7,   9,  10,  12,  14,  16],
        [ 19,  22,  25,  29,  34,  39,  44,  50],
        [ 57,  64,  72,  80,  89,  99, 108, 118],
        [128, 138, 148, 158, 167, 176, 184, 192],
        [199, 205, 211, 217, 221, 226, 229, 233],
        [236, 238, 240, 242, 244, 246, 247, 248],
        [249, 250, 250, 251, 251, 252, 252, 253]], dtype=np.uint8)

    result = exposure.adjust_sigmoid(image, 0.5, 10)
    assert_array_equal(result, expected)


def test_adjust_inv_sigmoid_cutoff_half():
    """Verifying the output with expected results for inverse sigmoid
    correction with cutoff equal to half and gain of 10"""
    image = np.arange(0, 255, 4, np.uint8).reshape(8,8)
    expected = np.array([[253, 253, 252, 252, 251, 251, 250, 249],
        [249, 248, 247, 245, 244, 242, 240, 238],
        [235, 232, 229, 225, 220, 215, 210, 204],
        [197, 190, 182, 174, 165, 155, 146, 136],
        [126, 116, 106,  96,  87,  78,  70,  62],
        [ 55,  49,  43,  37,  33,  28,  25,  21],
        [ 18,  16,  14,  12,  10,   8,   7,   6],
        [  5,   4,   4,   3,   3,   2,   2,   1]], dtype=np.uint8)

    result = exposure.adjust_sigmoid(image, 0.5, 10, True)
    assert_array_equal(result, expected)
