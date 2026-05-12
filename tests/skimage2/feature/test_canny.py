import pytest
import numpy as np
from numpy.testing import assert_equal
from scipy.ndimage import binary_dilation, binary_erosion

from _skimage2.feature import canny
import skimage as ski


class TestCanny:
    def test_00_00_zeros(self):
        '''Test that the Canny filter finds no points for a blank field'''
        image = np.zeros((20, 20))
        result = canny(
            image,
            sigma=4,
            low_threshold=0,
            high_threshold=0,
            mask=np.ones((20, 20), bool),
        )
        assert_equal(result, False)

    def test_00_01_zeros_mask(self):
        '''Test that the Canny filter finds no points in a masked image'''
        rng = np.random.default_rng(20260422)
        image = rng.uniform(size=(20, 20))
        result = canny(
            image,
            sigma=4,
            low_threshold=0,
            high_threshold=0,
            mask=np.zeros((20, 20), bool),
        )
        assert_equal(result, False)

    def test_01_01_circle(self):
        '''Test that the Canny filter finds the outlines of a circle'''
        i, j = np.mgrid[-200:200, -200:200].astype(float) / 200
        c = np.abs(np.sqrt(i * i + j * j) - 0.5) < 0.02
        result = canny(
            c.astype(float),
            sigma=4,
            low_threshold=0,
            high_threshold=0,
            mask=np.ones(c.shape, bool),
        )
        #
        # erode and dilate the circle to get rings that should contain the
        # outlines
        #
        cd = binary_dilation(c, iterations=3)
        ce = binary_erosion(c, iterations=3)
        cde = np.logical_and(cd, np.logical_not(ce))
        assert_equal(cde[result], True)
        #
        # The circle has a radius of 100. There are two rings here, one
        # for the inside edge and one for the outside. So that's
        # 100 * 2 * 2 * 3 for those places where pi is still 3.
        # The edge contains both pixels if there's a tie, so we
        # bump the count a little.
        point_count = np.sum(result)
        assert_equal(point_count > 1200, True)
        assert_equal(point_count < 1600, True)

    def test_01_02_circle_with_noise(self):
        '''Test that the Canny filter finds the circle outlines
        in a noisy image'''
        rng = np.random.default_rng(0)
        i, j = np.mgrid[-200:200, -200:200].astype(float) / 200
        c = np.abs(np.sqrt(i * i + j * j) - 0.5) < 0.02
        cf = c.astype(float) * 0.5 + rng.uniform(size=c.shape) * 0.5
        result = canny(
            cf,
            sigma=4,
            low_threshold=0.1,
            high_threshold=0.2,
            mask=np.ones(c.shape, bool),
        )
        #
        # erode and dilate the circle to get rings that should contain the
        # outlines
        #
        cd = binary_dilation(c, iterations=4)
        ce = binary_erosion(c, iterations=4)
        cde = np.logical_and(cd, np.logical_not(ce))
        assert_equal(cde[result], True)

        point_count = np.sum(result)
        assert_equal(point_count > 1200, True)
        assert_equal(point_count < 1600, True)

    def test_image_shape(self):
        image = np.zeros((20, 20, 20))
        with pytest.raises(ValueError, match=".*`image` must be a 2-dimensional"):
            canny(image)

    def test_mask_none(self):
        result1 = canny(
            np.zeros((20, 20)),
            sigma=4,
            low_threshold=0,
            high_threshold=0,
            mask=np.ones((20, 20), bool),
        )
        result2 = canny(np.zeros((20, 20)), sigma=4, low_threshold=0, high_threshold=0)
        assert_equal(result1, result2)

    def test_use_quantiles(self):
        image = ski.util.img_as_float(ski.data.camera()[::100, ::100])

        # Correct output produced manually with quantiles
        # of 0.8 and 0.6 for high and low respectively
        correct_output = np.array(
            [
                [False, False, False, False, False, False],
                [False, True, True, True, False, False],
                [False, False, False, True, False, False],
                [False, False, False, True, False, False],
                [False, False, True, True, False, False],
                [False, False, False, False, False, False],
            ]
        )

        result = canny(image, low_threshold=0.6, high_threshold=0.8, use_quantiles=True)

        assert_equal(result, correct_output)

    def test_img_all_ones(self):
        image = np.ones((10, 10))
        assert np.all(canny(image) == 0)

    def test_invalid_use_quantiles(self):
        image = ski.util.img_as_float(ski.data.camera()[::50, ::50])
        regex = r"Quantile thresholds must be between 0 and 1"

        with pytest.raises(ValueError, match=regex):
            canny(image, use_quantiles=True, low_threshold=0.5, high_threshold=3.6)

        with pytest.raises(ValueError, match=regex):
            canny(
                image,
                use_quantiles=True,
                low_threshold=-5,
                high_threshold=0.5,
            )

        with pytest.raises(ValueError, match=regex):
            canny(
                image,
                use_quantiles=True,
                low_threshold=99,
                high_threshold=0.9,
            )

        with pytest.raises(ValueError, match=regex):
            canny(image, use_quantiles=True, low_threshold=0.5, high_threshold=-100)

        # Example from issue #4282
        image = ski.data.camera()
        with pytest.raises(ValueError, match=regex):
            canny(
                image,
                use_quantiles=True,
                low_threshold=50,
                high_threshold=150,
            )

    def test_dtype(self):
        """Check that the same output is produced regardless of image dtype."""
        image_uint8 = ski.data.camera()
        image_float = ski.util.img_as_float(image_uint8)

        result_uint8 = canny(image_uint8)
        result_float = canny(image_float)

        assert_equal(result_uint8, result_float)

        low = 0.1
        high = 0.2

        result_float = canny(
            image_float, sigma=1.0, low_threshold=low, high_threshold=high
        )
        result_uint8 = canny(
            image_uint8, sigma=1.0, low_threshold=255 * low, high_threshold=255 * high
        )
        assert_equal(result_float, result_uint8)

    @pytest.mark.parametrize("mode", ['constant', 'nearest', 'reflect'])
    def test_full_mask_matches_no_mask(self, mode):
        """The masked and unmasked algorithms should return the same result."""
        image = ski.data.camera()
        result_none = canny(image, mode=mode)
        result_mask = canny(image, mode=mode, mask=np.ones_like(image, dtype=bool))
        assert_equal(result_none, result_mask)

    @pytest.mark.parametrize("dtype", [np.int64, np.uint64])
    def test_unsupported_int64(self, dtype):
        image = np.zeros((10, 10), dtype=dtype)
        image[3, 3] = np.iinfo(dtype).max
        with pytest.raises(
            ValueError, match=r"64-bit integer images are not supported"
        ):
            canny(image)

    @pytest.mark.parametrize(
        'mode, should_match',
        [
            ('nearest', True),
            ('reflect', False),
            ('constant', False),
            ('mirror', False),
            ('wrap', False),
        ],
    )
    def test_default_mode_is_nearest(self, mode, should_match):
        rng = np.random.default_rng(0)
        image = rng.random((64, 64))
        result_default = canny(image)
        result_other = canny(image, mode=mode)
        assert np.array_equal(result_default, result_other) == should_match
