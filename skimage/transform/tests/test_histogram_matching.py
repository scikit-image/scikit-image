import numpy as np

from skimage.transform import histogram_matching
from skimage import transform
from skimage import data

from skimage._shared.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal

import pytest


def test_get_separate_channels_single():
    # given
    image = np.ones((4, 5))
    expected_channels = np.ones((1, 4, 5))

    # when
    channels = histogram_matching._get_separate_channels(image)

    # then
    assert_array_equal(channels, expected_channels)


def test_get_separate_channels_multiple():
    # given
    image = np.full((4, 5, 3), [1, 0, 3])  # channels of zeros, ones and threes
    expected_channels = np.asarray([
        np.ones((4, 5)),
        np.zeros((4, 5)),
        np.full((4, 5), 3)
    ])

    # when
    channels = histogram_matching._get_separate_channels(image)

    # then
    assert_array_equal(channels, expected_channels)


@pytest.mark.parametrize('array, template, expected_array', [
    (np.arange(10), np.arange(100), np.arange(9,100,10)),
    (np.random.rand(4), np.ones(3), np.ones(4))
])
def test_match_array_values(array, template, expected_array):
    # when
    matched = histogram_matching._match_array_values(array, template)

    # then
    assert_array_almost_equal(matched, expected_array)


def _calculate_image_empirical_pdf(image):
    """Helper function for calculating empirical probability density function of a given image for all channels"""

    channels = histogram_matching._get_separate_channels(image)
    channels_pdf = []
    for channel in channels:
        channel_values, counts = np.unique(channel, return_counts=True)
        channel_quantiles = np.cumsum(counts).astype(np.float64)
        channel_quantiles /= channel_quantiles[-1]

        channels_pdf.append((channel_values, channel_quantiles))

    return np.asarray(channels_pdf)


image_rgb = data.chelsea()
template_rgb = data.astronaut()


@pytest.mark.parametrize('image, reference', [
    (image_rgb[:, :, 0], template_rgb[:, :, 0]),
    (image_rgb, template_rgb)
])
def test_match_histograms(image, reference):
    """Assert that pdf of matched image is close to the reference's pdf for all channels and all values of matched"""

    # when
    matched = transform.match_histograms(image, reference)

    matched_pdf = _calculate_image_empirical_pdf(matched)
    reference_pdf = _calculate_image_empirical_pdf(reference)

    # then
    for channel in range(len(matched_pdf)):
        reference_values, reference_quantiles = reference_pdf[channel]
        matched_values, matched_quantiles = matched_pdf[channel]

        for i, matched_value in enumerate(matched_values):
            closest_idx = (np.abs(reference_values - matched_value)).argmin()
            assert_almost_equal(matched_quantiles[i], reference_quantiles[closest_idx], decimal=2)


@pytest.mark.parametrize('image, reference', [
    (image_rgb, template_rgb[:, :, 0]),
    (image_rgb[:, :, 0], template_rgb)
])
def test_raises_value_error_on_channels_mismatch(image, reference):
    with pytest.raises(ValueError):
        transform.match_histograms(image, reference)
