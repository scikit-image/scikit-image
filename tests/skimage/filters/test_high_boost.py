"""Tests for ``skimage.filters.high_boost_filter``."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from skimage.filters import high_boost_filter


def _expected(image, boost_factor, sigma, channel_axis=None):
    working = np.array(image, dtype=image.dtype, copy=True)
    if channel_axis is None:
        sigmas = sigma
    else:
        axis = channel_axis % working.ndim
        sigmas = [sigma] * working.ndim
        sigmas[axis] = 0.0
    smoothed = gaussian_filter(working, sigma=sigmas, mode="nearest")
    factor = np.array(boost_factor, dtype=working.dtype)
    return working + factor * (working - smoothed)


class TestHighBoostFilter:
    def test_2d_matches_gaussian_high_boost(self):
        image = np.arange(16, dtype=np.float64).reshape(4, 4)
        output = high_boost_filter(image, boost_factor=1.5, sigma=1.0)
        expected = _expected(image, boost_factor=1.5, sigma=1.0)
        assert output.shape == (4, 4)
        assert output.dtype == np.float64
        np.testing.assert_allclose(output, expected)

    def test_3d_multichannel_channel_axis(self):
        image = np.arange(120, dtype=np.float64).reshape(4, 5, 6)
        output = high_boost_filter(
            image, boost_factor=2.0, sigma=0.8, channel_axis=-1
        )
        assert output.shape == image.shape
        for channel in range(image.shape[-1]):
            expected = high_boost_filter(
                image[..., channel], boost_factor=2.0, sigma=0.8
            )
            np.testing.assert_allclose(output[..., channel], expected)
        full = _expected(image, boost_factor=2.0, sigma=0.8, channel_axis=-1)
        np.testing.assert_allclose(output, full)

    def test_float32_dtype_preserved(self):
        image = np.linspace(0, 1, 32, dtype=np.float32).reshape(4, 8)
        output = high_boost_filter(image, boost_factor=1.25, sigma=1.0)
        assert output.dtype == np.float32
        assert output.shape == image.shape
        np.testing.assert_allclose(output, _expected(image, 1.25, 1.0))

        multichannel = np.linspace(0, 1, 48, dtype=np.float32).reshape(4, 4, 3)
        output_mc = high_boost_filter(
            multichannel, boost_factor=1.25, sigma=1.0, channel_axis=-1
        )
        assert output_mc.dtype == np.float32
        assert output_mc.shape == multichannel.shape

    def test_negative_boost_factor_raises(self):
        image = np.zeros((4, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="boost_factor"):
            high_boost_filter(image, boost_factor=-0.1)

    def test_negative_sigma_raises(self):
        image = np.zeros((4, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="sigma"):
            high_boost_filter(image, boost_factor=1.0, sigma=-1.0)

    def test_nan_input(self):
        image = np.ones((5, 5), dtype=np.float64)
        image[2, 2] = np.nan
        output = high_boost_filter(image, boost_factor=1.0, sigma=1.0)
        assert output.shape == image.shape
        assert np.isnan(output).all()

    def test_inf_input(self):
        image = np.ones((5, 5), dtype=np.float64)
        image[2, 2] = np.inf
        output = high_boost_filter(image, boost_factor=1.0, sigma=1.0)
        assert output.shape == image.shape
        assert not np.isfinite(output).all()
        assert np.isinf(output).any()

    def test_empty_array(self):
        image = np.zeros((0, 5), dtype=np.float32)
        output = high_boost_filter(image, boost_factor=1.0, sigma=1.0)
        assert output.shape == (0, 5)
        assert output.dtype == np.float32
        assert output.size == 0

        volume = np.zeros((2, 0, 3), dtype=np.float64)
        output_3d = high_boost_filter(
            volume, boost_factor=1.0, sigma=1.0, channel_axis=-1
        )
        assert output_3d.shape == volume.shape
        assert output_3d.dtype == np.float64

    def test_input_is_not_mutated(self):
        image = np.arange(9, dtype=np.float32).reshape(3, 3)
        original = image.copy()
        output = high_boost_filter(image, boost_factor=2.0, sigma=1.0)
        np.testing.assert_array_equal(image, original)
        assert output is not image
