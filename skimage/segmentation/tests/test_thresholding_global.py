import pytest
import numpy as np

import skimage as ski
from skimage.segmentation import threshold_li, threshold_otsu
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
