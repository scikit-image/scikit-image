import numpy as np
from numpy.testing import assert_array_almost_equal as assert_close

import skimage
from skimage import data
from skimage import exposure


# Test histogram equalization
# ===========================

# squeeze image intensities to lower image contrast
test_img = exposure.rescale_intensity(data.camera() / 5. + 100)


def test_equalize_ubyte():
    img = skimage.img_as_ubyte(test_img)
    img_eq = exposure.equalize(img)

    cdf, bin_edges = exposure.cumulative_distribution(img_eq)
    check_cdf_slope(cdf)


def test_equalize_float():
    img = skimage.img_as_float(test_img)
    img_eq = exposure.equalize(img)

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


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
