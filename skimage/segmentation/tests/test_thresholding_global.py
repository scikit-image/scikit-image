import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose

import skimage as ski
from skimage.segmentation import (
    threshold_li,
    threshold_otsu,
    threshold_yen,
    threshold_minimum,
    threshold_triangle,
    threshold_isodata,
)
from skimage.segmentation._thresholding_global import _cross_entropy
from skimage._shared.testing import assert_stacklevel


class Test_threshold_li:
    def test_simple(self):
        image = np.array(
            [
                [0, 0, 1, 3, 5],
                [0, 1, 4, 3, 4],
                [1, 2, 5, 4, 1],
                [2, 4, 5, 2, 1],
                [4, 5, 1, 0, 0],
            ],
            dtype=int,
        )
        assert 2 < threshold_li(image) < 3

    def test_negative_int(self):
        image = np.array(
            [
                [0, 0, 1, 3, 5],
                [0, 1, 4, 3, 4],
                [1, 2, 5, 4, 1],
                [2, 4, 5, 2, 1],
                [4, 5, 1, 0, 0],
            ],
            dtype=int,
        )
        image = image - 2
        assert 0 < threshold_li(image) < 1

    def test_float(self):
        image = np.array(
            [
                [0, 0, 1, 3, 5],
                [0, 1, 4, 3, 4],
                [1, 2, 5, 4, 1],
                [2, 4, 5, 2, 1],
                [4, 5, 1, 0, 0],
            ],
            dtype=float,
        )
        assert 2 < threshold_li(image) < 3

    def test_uniform(self):
        assert threshold_li(np.ones((10, 10))) == 1.0

    def test_camera(self):
        image = ski.util.img_as_ubyte(ski.data.camera())
        threshold = threshold_li(image)
        ce_actual = _cross_entropy(image, threshold=threshold)
        assert 78 < threshold_li(image) < 79
        assert ce_actual < _cross_entropy(image, threshold=threshold + 1)
        assert ce_actual < _cross_entropy(image, threshold=threshold - 1)

    def test_coins(self):
        image = ski.util.img_as_ubyte(ski.data.coins())
        threshold = threshold_li(image)
        ce_actual = _cross_entropy(image, threshold=threshold)
        assert 94 < threshold_li(image) < 95
        assert ce_actual < _cross_entropy(image, threshold=threshold + 1)
        # in the case of the coins image, the minimum cross-entropy is achieved one
        # threshold below that found by the iterative method. Not sure why that is
        # but `threshold_li` does find the stationary point of the function (ie the
        # tolerance can be reduced arbitrarily but the exact same threshold is
        # found), so my guess is some kind of histogram binning effect.
        assert ce_actual < _cross_entropy(image, threshold=threshold - 2)

    def test_coins_as_float(self):
        coins = ski.util.img_as_float(ski.data.coins())
        assert 94 / 255 < threshold_li(coins) < 95 / 255

    def test_astronaut(self):
        image = ski.util.img_as_ubyte(ski.data.astronaut())
        threshold = threshold_li(image)
        ce_actual = _cross_entropy(image, threshold=threshold)
        assert 64 < threshold < 65
        assert ce_actual < _cross_entropy(image, threshold=threshold + 1)
        assert ce_actual < _cross_entropy(image, threshold=threshold - 1)

    def test_nan(self):
        image = np.full((5, 5), np.nan)
        assert np.isnan(threshold_li(image))

    def test_inf_nan(self):
        image = np.array([np.inf, np.nan])
        assert threshold_li(image) == np.inf

    def test_minus_inf(self):
        image = np.array([np.inf, -np.inf])
        assert threshold_li(image) == 0

    def test_uniform_with_nan(self):
        image = np.array([8, 8, 8, 8, np.nan])
        assert threshold_li(image) == 8

    def test_arbitrary_start_point(self):
        cell = ski.data.cell()
        max_stationary_point = threshold_li(cell)
        low_stationary_point = threshold_li(cell, initial_guess=np.percentile(cell, 5))
        optimum = threshold_li(cell, initial_guess=np.percentile(cell, 95))
        assert 67 < max_stationary_point < 68
        assert 48 < low_stationary_point < 49
        assert 111 < optimum < 112

    def test_negative_initial_guess(self):
        coins = ski.data.coins()
        with pytest.raises(
            ValueError, match=".*initial guess.*must be within the range"
        ):
            threshold_li(coins, initial_guess=-5)

    @pytest.mark.parametrize(
        "image",
        [
            # See https://github.com/scikit-image/scikit-image/issues/4140
            [0, 0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0.1, 0, 0, 0.1, 0, 0.1],
            [0, 0, 0.1, 0, 0, 0.1, 0.01, 0.1],
            [0, 0, 1, 0, 0, 1, 0.5, 1],
            [1, 1],
            [1, 2],
            # See https://github.com/scikit-image/scikit-image/issues/6744
            [0, 254, 255],
            [0, 1, 255],
            [0.1, 0.8, 0.9],
        ],
    )
    def test_pathological(self, image):
        image = np.array(image)
        threshold = threshold_li(image)
        assert np.isfinite(threshold)


class Test_threshold_otsu:
    def test_simple(self):
        image = np.array(
            [
                [0, 0, 1, 3, 5],
                [0, 1, 4, 3, 4],
                [1, 2, 5, 4, 1],
                [2, 4, 5, 2, 1],
                [4, 5, 1, 0, 0],
            ],
            dtype=int,
        )
        assert threshold_otsu(image) == 2

    def test_negative_int(self):
        image = np.array(
            [
                [0, 0, 1, 3, 5],
                [0, 1, 4, 3, 4],
                [1, 2, 5, 4, 1],
                [2, 4, 5, 2, 1],
                [4, 5, 1, 0, 0],
            ],
            dtype=int,
        )
        image = image - 2
        assert threshold_otsu(image) == 0

    def test_float(self):
        image = np.array(
            [
                [0, 0, 1, 3, 5],
                [0, 1, 4, 3, 4],
                [1, 2, 5, 4, 1],
                [2, 4, 5, 2, 1],
                [4, 5, 1, 0, 0],
            ],
            dtype=float,
        )
        assert 2 <= threshold_otsu(image) < 3

    def test_camera(self):
        camera = ski.util.img_as_ubyte(ski.data.camera())
        assert 101 < threshold_otsu(camera) < 103

    def test_camera_histogram(self):
        camera = ski.util.img_as_ubyte(ski.data.camera())
        hist = ski.exposure.histogram(camera.ravel(), 256, source_range='image')
        assert 101 < threshold_otsu(hist=hist) < 103

    def test_camera_counts(self):
        camera = ski.util.img_as_ubyte(ski.data.camera())
        counts, bin_centers = ski.exposure.histogram(
            camera.ravel(), 256, source_range='image'
        )
        assert 101 < threshold_otsu(hist=counts) < 103

    def test_zero_count_histogram(self):
        """Issue #5497.

        As the histogram returned by np.bincount starts with zero,
        it resulted in NaN-related issues.
        """
        x = np.array([1, 2])

        t1 = threshold_otsu(x)
        t2 = threshold_otsu(hist=np.bincount(x))
        assert t1 == t2

    def test_coins(self):
        coins = ski.util.img_as_ubyte(ski.data.coins())
        assert 106 < threshold_otsu(coins) < 108

    def test_coins_as_float(self):
        coins = ski.util.img_as_float(ski.data.coins())
        assert 0.41 < threshold_otsu(coins) < 0.42

    def test_astronaut(self):
        img = ski.util.img_as_ubyte(ski.data.astronaut())
        regex = ".*expected to work correctly only for grayscale images"
        with pytest.warns(UserWarning, match=regex) as record:
            assert 109 < threshold_otsu(img) < 111
        assert_stacklevel(record)
        assert len(record) == 1

    def test_uniform(self):
        img = np.ones((10, 10), dtype=np.uint8)
        assert threshold_otsu(img) == 1

    def test_uniform_3d(self):
        img = np.ones((10, 10, 10), dtype=np.uint8)
        assert threshold_otsu(img) == 1


class Test_threshold_yen:
    def test_simple(self):
        image = np.array(
            [
                [0, 0, 1, 3, 5],
                [0, 1, 4, 3, 4],
                [1, 2, 5, 4, 1],
                [2, 4, 5, 2, 1],
                [4, 5, 1, 0, 0],
            ],
            dtype=int,
        )
        assert threshold_yen(image) == 2

    def test_negative_int(self):
        image = np.array(
            [
                [0, 0, 1, 3, 5],
                [0, 1, 4, 3, 4],
                [1, 2, 5, 4, 1],
                [2, 4, 5, 2, 1],
                [4, 5, 1, 0, 0],
            ],
            dtype=int,
        )
        image = image - 2
        assert threshold_yen(image) == 0

    def test_float(self):
        image = np.array(
            [
                [0, 0, 1, 3, 5],
                [0, 1, 4, 3, 4],
                [1, 2, 5, 4, 1],
                [2, 4, 5, 2, 1],
                [4, 5, 1, 0, 0],
            ],
            dtype=float,
        )
        assert 2 <= threshold_yen(image) < 3

    def test_arange(self):
        image = np.arange(256)
        assert threshold_yen(image) == 127

    def test_uint8(self):
        image = np.zeros([2, 256], dtype=np.uint8)
        image[0] = 255
        assert threshold_yen(image) < 1

    def test_blank_zero(self):
        image = np.zeros((5, 5), dtype=np.uint8)
        assert threshold_yen(image) == 0

    def test_blank_max(self):
        image = np.empty((5, 5), dtype=np.uint8)
        image.fill(255)
        assert threshold_yen(image) == 255

    def test_camera(self):
        camera = ski.util.img_as_ubyte(ski.data.camera())
        assert 145 < threshold_yen(camera) < 147

    def test_camera_histogram(self):
        camera = ski.util.img_as_ubyte(ski.data.camera())
        hist = ski.exposure.histogram(camera.ravel(), 256, source_range='image')
        assert 145 < threshold_yen(hist=hist) < 147

    def test_camera_counts(self):
        camera = ski.util.img_as_ubyte(ski.data.camera())
        counts, bin_centers = ski.exposure.histogram(
            camera.ravel(), 256, source_range='image'
        )
        assert 145 < threshold_yen(hist=counts) < 147

    def test_coins(self):
        coins = ski.util.img_as_ubyte(ski.data.coins())
        assert 109 < threshold_yen(coins) < 111

    def test_coins_as_float(self):
        coins = ski.util.img_as_float(ski.data.coins())
        assert 0.43 < threshold_yen(coins) < 0.44


class Test_threshold_minimum:
    def test_minimum(self):
        image = np.array(
            [
                [0, 0, 1, 3, 5],
                [0, 1, 4, 3, 4],
                [1, 2, 5, 4, 1],
                [2, 4, 5, 2, 1],
                [4, 5, 1, 0, 0],
            ],
            dtype=int,
        )
        with pytest.raises(
            RuntimeError, match="Unable to find two maxima in histogram"
        ):
            threshold_minimum(image)

    def test_camera(self):
        camera = ski.util.img_as_ubyte(ski.data.camera())

        threshold = threshold_minimum(camera)
        assert threshold == 85

    def test_astronaut(self):
        astronaut = ski.util.img_as_ubyte(ski.data.astronaut())
        threshold = threshold_minimum(astronaut)
        assert threshold == 114

    def test_camera_histogram(self):
        camera = ski.util.img_as_ubyte(ski.data.camera())
        hist = ski.exposure.histogram(camera.ravel(), 256, source_range='image')
        threshold = threshold_minimum(hist=hist)
        assert threshold == 85

    def test_camera_counts(self):
        camera = ski.util.img_as_ubyte(ski.data.camera())
        counts, bin_centers = ski.exposure.histogram(
            camera.ravel(), 256, source_range='image'
        )
        threshold = threshold_minimum(hist=counts)
        assert threshold == 85

    def test_synthetic(self):
        img = np.arange(25 * 25, dtype=np.uint8).reshape((25, 25))
        img[0:9, :] = 50
        img[14:25, :] = 250
        threshold = threshold_minimum(img)
        assert threshold == 95

    def test_failure(self):
        img = np.zeros((16 * 16), dtype=np.uint8)
        with pytest.raises(RuntimeError):
            threshold_minimum(img)


class Test_threshold_triangle:
    @pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float16, np.float32])
    def test_uniform_images(self, dtype):
        assert threshold_triangle(np.zeros((10, 10), dtype=dtype)) == 0
        assert threshold_triangle(np.ones((10, 10), dtype=dtype)) == 1
        assert threshold_triangle(np.full((10, 10), 2, dtype=dtype)) == 2

    def test_uint_images(self):
        assert threshold_triangle(np.invert(ski.data.text())) == 151
        assert threshold_triangle(ski.data.text()) == 104
        assert threshold_triangle(ski.data.coins()) == 80
        assert threshold_triangle(np.invert(ski.data.coins())) == 175

    def test_float_images(self):
        text = ski.data.text()
        int_bins = text.max() - text.min() + 1
        # Set nbins to match the uint case and threshold as float.
        assert round(threshold_triangle(text.astype(float), nbins=int_bins)) == 104
        # Check that rescaling image to floats in unit interval is equivalent.
        assert round(threshold_triangle(text / 255.0, nbins=int_bins) * 255) == 104
        # Repeat for inverted image.
        assert (
            round(threshold_triangle(np.invert(text).astype(float), nbins=int_bins))
            == 151
        )
        assert (
            round(threshold_triangle(np.invert(text) / 255.0, nbins=int_bins) * 255)
            == 151
        )

    def test_flip(self):
        # Depending on the skewness, the algorithm flips the histogram.
        # We check that the flip doesn't affect too much the result.
        img = ski.data.camera()
        inv_img = np.invert(img)
        t = threshold_triangle(inv_img)
        t_inv_img = inv_img > t
        t_inv_inv_img = np.invert(t_inv_img)

        t = threshold_triangle(img)
        t_img = img > t

        # Check that most of the pixels are identical
        # See numpy #7685 for a future np.testing API
        unequal_pos = np.where(t_img.ravel() != t_inv_inv_img.ravel())
        assert len(unequal_pos[0]) / t_img.size < 1e-2


class Test_threshold_isodata:
    def test_simple(self):
        image = np.array(
            [
                [0, 0, 1, 3, 5],
                [0, 1, 4, 3, 4],
                [1, 2, 5, 4, 1],
                [2, 4, 5, 2, 1],
                [4, 5, 1, 0, 0],
            ],
            dtype=int,
        )
        assert threshold_isodata(image.copy()) == 2
        assert threshold_isodata(image.copy(), return_all=True) == [2]

    def test_blank_zero(self):
        image = np.zeros((5, 5), dtype=np.uint8)
        assert threshold_isodata(image) == 0
        assert threshold_isodata(image, return_all=True) == [0]

    def test_linspace(self):
        image = np.linspace(-127, 0, 256)
        assert -63.8 < threshold_isodata(image) < -63.6
        assert_allclose(
            threshold_isodata(image, return_all=True), [-63.74804688, -63.25195312]
        )

    def test_16bit(self):
        np.random.seed(0)
        imfloat = np.random.rand(256, 256)
        assert 0.49 < threshold_isodata(imfloat, nbins=1024) < 0.51
        assert all(0.49 < threshold_isodata(imfloat, nbins=1024, return_all=True))

    def test_camera_image(self):
        camera = ski.util.img_as_ubyte(ski.data.camera())

        threshold = threshold_isodata(camera)
        assert (
            np.floor(
                (camera[camera <= threshold].mean() + camera[camera > threshold].mean())
                / 2.0
            )
            == threshold
        )
        assert threshold == 102

        assert (threshold_isodata(camera, return_all=True) == [102, 103]).all()

    def test_camera_image_histogram(self):
        camera = ski.util.img_as_ubyte(ski.data.camera())
        hist = ski.exposure.histogram(camera.ravel(), 256, source_range='image')
        threshold = threshold_isodata(hist=hist)
        assert threshold == 102

    def test_camera_image_counts(self):
        camera = ski.util.img_as_ubyte(ski.data.camera())
        counts, bin_centers = ski.exposure.histogram(
            camera.ravel(), 256, source_range='image'
        )
        threshold = threshold_isodata(hist=counts)
        assert threshold == 102

    def test_coins_image(self):
        coins = ski.util.img_as_ubyte(ski.data.coins())

        threshold = threshold_isodata(coins)
        assert (
            np.floor(
                (coins[coins <= threshold].mean() + coins[coins > threshold].mean())
                / 2.0
            )
            == threshold
        )
        assert threshold == 107

        assert threshold_isodata(coins, return_all=True) == [107]

    def test_moon_image(self):
        moon = ski.util.img_as_ubyte(ski.data.moon())

        threshold = threshold_isodata(moon)
        assert (
            np.floor(
                (moon[moon <= threshold].mean() + moon[moon > threshold].mean()) / 2.0
            )
            == threshold
        )
        assert threshold == 86

        thresholds = threshold_isodata(moon, return_all=True)
        for threshold in thresholds:
            assert (
                np.floor(
                    (moon[moon <= threshold].mean() + moon[moon > threshold].mean())
                    / 2.0
                )
                == threshold
            )
        assert_equal(thresholds, [86, 87, 88, 122, 123, 124, 139, 140])

    def test_moon_image_negative_int(self):
        moon = ski.util.img_as_ubyte(ski.data.moon()).astype(np.int32)
        moon -= 100

        threshold = threshold_isodata(moon)
        assert (
            np.floor(
                (moon[moon <= threshold].mean() + moon[moon > threshold].mean()) / 2.0
            )
            == threshold
        )
        assert threshold == -14

        thresholds = threshold_isodata(moon, return_all=True)
        for threshold in thresholds:
            assert (
                np.floor(
                    (moon[moon <= threshold].mean() + moon[moon > threshold].mean())
                    / 2.0
                )
                == threshold
            )
        assert_equal(thresholds, [-14, -13, -12, 22, 23, 24, 39, 40])

    def test_moon_image_negative_float(self):
        moon = ski.util.img_as_ubyte(ski.data.moon()).astype(np.float64)
        moon -= 100

        assert -14 < threshold_isodata(moon) < -13

        thresholds = threshold_isodata(moon, return_all=True)
        assert_allclose(
            thresholds,
            [
                -13.83789062,
                -12.84179688,
                -11.84570312,
                22.02148438,
                23.01757812,
                24.01367188,
                38.95507812,
                39.95117188,
            ],
        )
