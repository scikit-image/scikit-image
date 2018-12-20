import pytest
import numpy as np
from scipy import ndimage as ndi
import skimage
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage.filters.thresholding import (threshold_local,
                                          threshold_otsu,
                                          threshold_li,
                                          threshold_yen,
                                          threshold_isodata,
                                          threshold_niblack,
                                          threshold_sauvola,
                                          threshold_mean,
                                          threshold_triangle,
                                          threshold_minimum,
                                          try_all_threshold,
                                          _mean_std,
                                          _cross_entropy)
from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_almost_equal
from skimage._shared.testing import assert_array_equal


class TestSimpleImage():
    def setup(self):
        self.image = np.array([[0, 0, 1, 3, 5],
                               [0, 1, 4, 3, 4],
                               [1, 2, 5, 4, 1],
                               [2, 4, 5, 2, 1],
                               [4, 5, 1, 0, 0]], dtype=int)

    def test_minimum(self):
        with pytest.raises(RuntimeError):
            threshold_minimum(self.image)

    def test_try_all_threshold(self):
        fig, ax = try_all_threshold(self.image)
        all_texts = [axis.texts for axis in ax if axis.texts != []]
        text_content = [text.get_text() for x in all_texts for text in x]
        assert 'RuntimeError' in text_content

    def test_otsu(self):
        assert threshold_otsu(self.image) == 2

    def test_otsu_negative_int(self):
        image = self.image - 2
        assert threshold_otsu(image) == 0

    def test_otsu_float_image(self):
        image = np.float64(self.image)
        assert 2 <= threshold_otsu(image) < 3

    def test_li(self):
        assert 2 < threshold_li(self.image) < 3

    def test_li_negative_int(self):
        image = self.image - 2
        assert 0 < threshold_li(image) < 1

    def test_li_float_image(self):
        image = self.image.astype(float)
        assert 2 < threshold_li(image) < 3

    def test_li_constant_image(self):
        assert threshold_li(np.ones((10, 10))) == 1.

    def test_yen(self):
        assert threshold_yen(self.image) == 2

    def test_yen_negative_int(self):
        image = self.image - 2
        assert threshold_yen(image) == 0

    def test_yen_float_image(self):
        image = np.float64(self.image)
        assert 2 <= threshold_yen(image) < 3

    def test_yen_arange(self):
        image = np.arange(256)
        assert threshold_yen(image) == 127

    def test_yen_binary(self):
        image = np.zeros([2, 256], dtype=np.uint8)
        image[0] = 255
        assert threshold_yen(image) < 1

    def test_yen_blank_zero(self):
        image = np.zeros((5, 5), dtype=np.uint8)
        assert threshold_yen(image) == 0

    def test_yen_blank_max(self):
        image = np.empty((5, 5), dtype=np.uint8)
        image.fill(255)
        assert threshold_yen(image) == 255

    def test_isodata(self):
        assert threshold_isodata(self.image) == 2
        assert threshold_isodata(self.image, return_all=True) == [2]

    def test_isodata_blank_zero(self):
        image = np.zeros((5, 5), dtype=np.uint8)
        assert threshold_isodata(image) == 0
        assert threshold_isodata(image, return_all=True) == [0]

    def test_isodata_linspace(self):
        image = np.linspace(-127, 0, 256)
        assert -63.8 < threshold_isodata(image) < -63.6
        assert_almost_equal(threshold_isodata(image, return_all=True),
                            [-63.74804688, -63.25195312])

    def test_isodata_16bit(self):
        np.random.seed(0)
        imfloat = np.random.rand(256, 256)
        assert 0.49 < threshold_isodata(imfloat, nbins=1024) < 0.51
        assert all(0.49 < threshold_isodata(imfloat, nbins=1024,
                                            return_all=True))

    def test_threshold_local_gaussian(self):
        ref = np.array(
            [[False, False, False, False,  True],
             [False, False,  True, False,  True],
             [False, False,  True,  True, False],
             [False,  True,  True, False, False],
             [ True,  True, False, False, False]]
        )
        out = threshold_local(self.image, 3, method='gaussian')
        assert_equal(ref, self.image > out)

        out = threshold_local(self.image, 3, method='gaussian',
                              param=1./3.)
        assert_equal(ref, self.image > out)

    def test_threshold_local_mean(self):
        ref = np.array(
            [[False, False, False, False,  True],
             [False, False,  True, False,  True],
             [False, False,  True,  True, False],
             [False,  True,  True, False, False],
             [ True,  True, False, False, False]]
        )
        out = threshold_local(self.image, 3, method='mean')
        assert_equal(ref, self.image > out)

    def test_threshold_local_median(self):
        ref = np.array(
            [[False, False, False, False,  True],
             [False, False,  True, False, False],
             [False, False,  True, False, False],
             [False, False,  True,  True, False],
             [False,  True, False, False, False]]
        )
        out = threshold_local(self.image, 3, method='median')
        assert_equal(ref, self.image > out)

    def test_threshold_local_median_constant_mode(self):
        out = threshold_local(self.image, 3, method='median',
                              mode='constant', cval=20)
        expected = np.array([[20.,  1.,  3.,  4., 20.],
                            [ 1.,  1.,  3.,  4.,  4.],
                            [ 2.,  2.,  4.,  4.,  4.],
                            [ 4.,  4.,  4.,  1.,  2.],
                            [20.,  5.,  5.,  2., 20.]])
        assert_equal(expected, out)

    def test_threshold_niblack(self):
        ref = np.array(
            [[False, False, False, True, True],
             [False, True, True, True, True],
             [False, True, True, True, False],
             [False, True, True, True, True],
             [True, True, False, False, False]]
        )
        thres = threshold_niblack(self.image, window_size=3, k=0.5)
        out = self.image > thres
        assert_equal(ref, out)

    def test_threshold_sauvola(self):
        ref = np.array(
            [[False, False, False, True, True],
             [False, False, True, True, True],
             [False, False, True, True, False],
             [False, True, True, True, False],
             [True, True, False, False, False]]
        )
        thres = threshold_sauvola(self.image, window_size=3, k=0.2, r=128)
        out = self.image > thres
        assert_equal(ref, out)

    def test_threshold_niblack_iterable_window_size(self):
        ref = np.array(
            [[False, False, False, True, True],
             [False, False, True, True, True],
             [False, True, True, True, False],
             [False, True, True, True, False],
             [True, True, False, False, False]]
        )
        thres = threshold_niblack(self.image, window_size=[3, 5], k=0.5)
        out = self.image > thres
        assert_array_equal(ref, out)

    def test_threshold_sauvola_iterable_window_size(self):
        ref = np.array(
            [[False, False, False, True, True],
             [False, False, True, True, True],
             [False, False, True, True, False],
             [False, True, True, True, False],
             [True, True, False, False, False]]
        )
        thres = threshold_sauvola(self.image, window_size=(3, 5), k=0.2, r=128)
        out = self.image > thres
        assert_array_equal(ref, out)


def test_otsu_camera_image():
    camera = skimage.img_as_ubyte(data.camera())
    assert 86 < threshold_otsu(camera) < 88


def test_otsu_coins_image():
    coins = skimage.img_as_ubyte(data.coins())
    assert 106 < threshold_otsu(coins) < 108


def test_otsu_coins_image_as_float():
    coins = skimage.img_as_float(data.coins())
    assert 0.41 < threshold_otsu(coins) < 0.42


def test_otsu_astro_image():
    img = skimage.img_as_ubyte(data.astronaut())
    with expected_warnings(['grayscale']):
        assert 109 < threshold_otsu(img) < 111


def test_otsu_one_color_image():
    img = np.ones((10, 10), dtype=np.uint8)
    with testing.raises(ValueError):
        threshold_otsu(img)


def test_li_camera_image():
    image = skimage.img_as_ubyte(data.camera())
    threshold = threshold_li(image)
    ce_actual = _cross_entropy(image, threshold)
    assert 62 < threshold_li(image) < 63
    assert ce_actual < _cross_entropy(image, threshold + 1)
    assert ce_actual < _cross_entropy(image, threshold - 1)


def test_li_coins_image():
    image = skimage.img_as_ubyte(data.coins())
    threshold = threshold_li(image)
    ce_actual = _cross_entropy(image, threshold)
    assert 94 < threshold_li(image) < 95
    assert ce_actual < _cross_entropy(image, threshold + 1)
    # in the case of the coins image, the minimum cross-entropy is achieved one
    # threshold below that found by the iterative method. Not sure why that is
    # but `threshold_li` does find the stationary point of the function (ie the
    # tolerance can be reduced arbitrarily but the exact same threshold is
    # found), so my guess some kind of histogram binning effect.
    assert ce_actual < _cross_entropy(image, threshold - 2)


def test_li_coins_image_as_float():
    coins = skimage.img_as_float(data.coins())
    assert 94/255 < threshold_li(coins) < 95/255


def test_li_astro_image():
    image = skimage.img_as_ubyte(data.astronaut())
    threshold = threshold_li(image)
    ce_actual = _cross_entropy(image, threshold)
    assert 64 < threshold < 65
    assert ce_actual < _cross_entropy(image, threshold + 1)
    assert ce_actual < _cross_entropy(image, threshold - 1)


def test_li_nan_image():
    image = np.full((5, 5), np.nan)
    assert np.isnan(threshold_li(image))


def test_li_inf_image():
    image = np.array([np.inf, np.nan])
    assert threshold_li(image) == np.inf


def test_li_inf_minus_inf():
    image = np.array([np.inf, -np.inf])
    assert threshold_li(image) == 0


def test_li_constant_image_with_nan():
    image = np.array([8, 8, 8, 8, np.nan])
    assert threshold_li(image) == 8


def test_yen_camera_image():
    camera = skimage.img_as_ubyte(data.camera())
    assert 197 < threshold_yen(camera) < 199


def test_yen_coins_image():
    coins = skimage.img_as_ubyte(data.coins())
    assert 109 < threshold_yen(coins) < 111


def test_yen_coins_image_as_float():
    coins = skimage.img_as_float(data.coins())
    assert 0.43 < threshold_yen(coins) < 0.44


def test_local_even_block_size_error():
    img = data.camera()
    with testing.raises(ValueError):
        threshold_local(img, block_size=4)


def test_isodata_camera_image():
    camera = skimage.img_as_ubyte(data.camera())

    threshold = threshold_isodata(camera)
    assert np.floor((camera[camera <= threshold].mean() +
                     camera[camera > threshold].mean()) / 2.0) == threshold
    assert threshold == 87

    assert threshold_isodata(camera, return_all=True) == [87]


def test_isodata_coins_image():
    coins = skimage.img_as_ubyte(data.coins())

    threshold = threshold_isodata(coins)
    assert np.floor((coins[coins <= threshold].mean() +
                     coins[coins > threshold].mean()) / 2.0) == threshold
    assert threshold == 107

    assert threshold_isodata(coins, return_all=True) == [107]


def test_isodata_moon_image():
    moon = skimage.img_as_ubyte(data.moon())

    threshold = threshold_isodata(moon)
    assert np.floor((moon[moon <= threshold].mean() +
                     moon[moon > threshold].mean()) / 2.0) == threshold
    assert threshold == 86

    thresholds = threshold_isodata(moon, return_all=True)
    for threshold in thresholds:
        assert np.floor((moon[moon <= threshold].mean() +
                         moon[moon > threshold].mean()) / 2.0) == threshold
    assert_equal(thresholds, [86, 87, 88, 122, 123, 124, 139, 140])


def test_isodata_moon_image_negative_int():
    moon = skimage.img_as_ubyte(data.moon()).astype(np.int32)
    moon -= 100

    threshold = threshold_isodata(moon)
    assert np.floor((moon[moon <= threshold].mean() +
                     moon[moon > threshold].mean()) / 2.0) == threshold
    assert threshold == -14

    thresholds = threshold_isodata(moon, return_all=True)
    for threshold in thresholds:
        assert np.floor((moon[moon <= threshold].mean() +
                         moon[moon > threshold].mean()) / 2.0) == threshold
    assert_equal(thresholds, [-14, -13, -12,  22,  23,  24,  39,  40])


def test_isodata_moon_image_negative_float():
    moon = skimage.img_as_ubyte(data.moon()).astype(np.float64)
    moon -= 100

    assert -14 < threshold_isodata(moon) < -13

    thresholds = threshold_isodata(moon, return_all=True)
    assert_almost_equal(thresholds,
                        [-13.83789062, -12.84179688, -11.84570312, 22.02148438,
                         23.01757812, 24.01367188, 38.95507812, 39.95117188])


def test_threshold_minimum():
    camera = skimage.img_as_ubyte(data.camera())

    threshold = threshold_minimum(camera)
    assert_equal(threshold, 76)

    astronaut = skimage.img_as_ubyte(data.astronaut())
    threshold = threshold_minimum(astronaut)
    assert_equal(threshold, 114)


def test_threshold_minimum_synthetic():
    img = np.arange(25*25, dtype=np.uint8).reshape((25, 25))
    img[0:9, :] = 50
    img[14:25, :] = 250

    threshold = threshold_minimum(img)
    assert_equal(threshold, 95)


def test_threshold_minimum_failure():
    img = np.zeros((16*16), dtype=np.uint8)
    with testing.raises(RuntimeError):
        threshold_minimum(img)


def test_mean():
    img = np.zeros((2, 6))
    img[:, 2:4] = 1
    img[:, 4:] = 2
    assert(threshold_mean(img) == 1.)


def test_triangle_uint_images():
    assert(threshold_triangle(np.invert(data.text())) == 151)
    assert(threshold_triangle(data.text()) == 104)
    assert(threshold_triangle(data.coins()) == 80)
    assert(threshold_triangle(np.invert(data.coins())) == 175)


def test_triangle_float_images():
    text = data.text()
    int_bins = text.max() - text.min() + 1
    # Set nbins to match the uint case and threshold as float.
    assert(round(threshold_triangle(
        text.astype(np.float), nbins=int_bins)) == 104)
    # Check that rescaling image to floats in unit interval is equivalent.
    assert(round(threshold_triangle(text / 255., nbins=int_bins) * 255) == 104)
    # Repeat for inverted image.
    assert(round(threshold_triangle(
        np.invert(text).astype(np.float), nbins=int_bins)) == 151)
    assert (round(threshold_triangle(
        np.invert(text) / 255., nbins=int_bins) * 255) == 151)


def test_triangle_flip():
    # Depending on the skewness, the algorithm flips the histogram.
    # We check that the flip doesn't affect too much the result.
    img = data.camera()
    inv_img = np.invert(img)
    t = threshold_triangle(inv_img)
    t_inv_img = inv_img > t
    t_inv_inv_img = np.invert(t_inv_img)

    t = threshold_triangle(img)
    t_img = img > t

    # Check that most of the pixels are identical
    # See numpy #7685 for a future np.testing API
    unequal_pos = np.where(t_img.ravel() != t_inv_inv_img.ravel())
    assert(len(unequal_pos[0]) / t_img.size < 1e-2)


@pytest.mark.parametrize(
    "window_size, mean_kernel",
    [(11, np.full((11,) * 2,  1 / 11 ** 2)),
     ((11, 11), np.full((11, 11), 1 / 11 ** 2)),
     ((9, 13), np.full((9, 13), 1 / np.prod((9, 13)))),
     ((13, 9), np.full((13, 9), 1 / np.prod((13, 9)))),
     ((1, 9), np.full((1, 9), 1 / np.prod((1, 9))))
     ]
)
def test_mean_std_2d(window_size, mean_kernel):
    image = np.random.rand(256, 256)
    m, s = _mean_std(image, w=window_size)
    expected_m = ndi.convolve(image, mean_kernel, mode='mirror')
    np.testing.assert_allclose(m, expected_m)
    expected_s = ndi.generic_filter(image, np.std, size=window_size,
                                    mode='mirror')
    np.testing.assert_allclose(s, expected_s)


@pytest.mark.parametrize(
    "window_size, mean_kernel", [
        (5, np.full((5,) * 3, 1 / 5) ** 3),
        ((5, 5, 5), np.full((5, 5, 5), 1 / 5 ** 3)),
        ((1, 5, 5), np.full((1, 5, 5), 1 / 5 ** 2)),
        ((3, 5, 7), np.full((3, 5, 7), 1 / np.prod((3, 5, 7))))]
)
def test_mean_std_3d(window_size, mean_kernel):
    image = np.random.rand(40, 40, 40)
    m, s = _mean_std(image, w=window_size)
    expected_m = ndi.convolve(image, mean_kernel, mode='mirror')
    np.testing.assert_allclose(m, expected_m)
    expected_s = ndi.generic_filter(image, np.std, size=window_size,
                                    mode='mirror')
    np.testing.assert_allclose(s, expected_s)


def test_niblack_sauvola_pathological_image():
    # For certain values, floating point error can cause
    # E(X^2) - (E(X))^2 to be negative, and taking the square root of this
    # resulted in NaNs. Here we check that these are safely caught.
    # see https://github.com/scikit-image/scikit-image/issues/3007
    value = 0.03082192 + 2.19178082e-09
    src_img = np.full((4, 4), value).astype(np.float64)
    assert not np.any(np.isnan(threshold_niblack(src_img)))
