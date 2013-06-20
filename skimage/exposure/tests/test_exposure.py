import warnings

import numpy as np
from numpy.testing import assert_array_almost_equal as assert_close
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

def test_gamma_correct_one():
    """Same image should be returned for gamma equal to one"""
    image = data.camera()
    result = exposure.correct(image, 'gamma', 1)
    assert result.mean() == image.mean()
    assert result.std() == image.std()


def test_gamma_correct_zero():
    """White image should be returned for gamma equal to zero"""
    image = data.camera()
    result = exposure.correct(image, 'gamma', 0)
    dtype = image.dtype.type
    assert result.mean() == dtype_range[dtype][1]
    assert result.std() == 0


def test_gamma_correct_less_one():
    """Output's mean should be greater than input's mean for gamma less than
    one"""
    image = data.camera()
    result = exposure.correct(image, 'gamma', 0.5)
    assert result.mean() > image.mean()


def test_gamma_correct_greater_one():
    """Output's mean should be less than input's mean for gamma greater than
    one"""
    image = data.camera()
    result = exposure.correct(image,'gamma', 2)
    assert result.mean() < image.mean()


# Test Logarithmic Correction
# ===========================

def test_logarithmic_correct():
    """Output's mean should be greater than input's mean for logarithmic
    correction with multiplier constant equal to unity"""
    image = data.camera()
    result = exposure.correct(image, 'logarithmic')
    assert result.mean() > image.mean()


def test_inv_logarithmic_correct():
    """Output's mean should be less than input's mean for inverse logarithmic
    correction with multiplier constant equal to unity"""
    image = data.camera()
    result = exposure.correct(image, 'logarithmic', -1)
    assert result.mean() < image.mean()


# Test Sigmoid Correction
# =======================

def test_sigmoid_correct_cutoff_one():
    """Output's mean should be less than input's mean for sigmoid
    correction with cutoff equal to one and gain of 10"""
    image = data.camera()
    result = exposure.correct(image, 'sigmoid', 10, 1)
    assert result.mean() < image.mean()


def test_sigmoid_correct_cutoff_zero():
    """Output's mean should be greater than input's mean for sigmoid
    correction with cutoff equal to zero and gain of 10"""
    image = data.camera()
    result = exposure.correct(image, 'sigmoid', 10, 0)
    assert result.mean() > image.mean()
