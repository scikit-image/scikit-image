import numpy as np

import skimage
from skimage import data
from skimage import exposure


# squeeze image intensities to lower image contrast
test_img = data.camera() / 5 + 100


def test_equalize_hist_ubyte():
    img_eq = exposure.equalize_hist(test_img)

    cdf, bin_edges = exposure.cumulative_distribution(img_eq)
    check_cdf_slope(cdf)


def test_equalize_hist_float():
    img = skimage.img_as_float(test_img)
    img_eq = exposure.equalize_hist(img)

    cdf, bin_edges = exposure.cumulative_distribution(img_eq)
    check_cdf_slope(cdf)


def check_cdf_slope(cdf):
    """Slope of cdf which should equal 1 for an equalized histogram."""
    norm_intensity = np.linspace(0, 1, len(cdf))
    slope, intercept = np.polyfit(norm_intensity, cdf, 1)
    assert 0.9 < slope < 1.1


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()

