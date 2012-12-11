import numpy as np
from numpy.testing import assert_array_almost_equal as assert_close
import skimage
from skimage import data
from skimage import exposure
from skimage.color import rgb2gray
from skimage.util.dtype import dtype_range
from skimage.io import use_plugin

use_plugin('matplotlib')

# Test histogram equalization
# ===========================

# squeeze image intensities to lower image contrast
test_img = exposure.rescale_intensity(data.camera() / 5. + 100)


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
    assert_almost_equal(peak_snr(full_scale, adapted), 28.6262034)
    assert_almost_equal(norm_brightness_err(full_scale, adapted),
                        0.0410010)
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
    assert_almost_equal(peak_snr(img, adapted), 77.58419, 5)
    assert_almost_equal(norm_brightness_err(img, adapted), 0.0376632)
    return data, adapted


def test_adapthist_color():
    '''Test a color uint16 image
    '''
    img = skimage.img_as_uint(data.lena())
    adapted = exposure.equalize_adapthist(img, clip_limit=0.01)
    assert_almost_equal = np.testing.assert_almost_equal
    assert adapted.min() == 0
    assert adapted.max() == 1.0
    assert img.shape == adapted.shape
    full_scale = skimage.exposure.rescale_intensity(img)
    assert_almost_equal(peak_snr(full_scale, adapted), 64.7167515)
    assert_almost_equal(norm_brightness_err(full_scale, adapted),
                        0.17922429)
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
