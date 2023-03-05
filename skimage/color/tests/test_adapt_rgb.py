# Importing necessary libraries
from functools import partial

import numpy as np

from skimage import img_as_float, img_as_uint
from skimage import color, data, filters
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value

# Down-sample image for quicker testing.
COLOR_IMAGE = data.astronaut()[::5, ::6]
GRAY_IMAGE = data.camera()[::5, ::5]

SIGMA = 3
smooth = partial(filters.gaussian, sigma=SIGMA)
assert_allclose = partial(np.testing.assert_allclose, atol=1e-8)


@adapt_rgb(each_channel)
def edges_each(image):
    """
    Applies Sobel edge detection to an RGB image, using each color channel
    independently.

    Parameters
    ----------
    image : ndarray
        Input RGB image.

    Returns
    -------
    result : ndarray
        Edge map of the input image, with each color channel processed
        independently.
    """
    return filters.sobel(image)


@adapt_rgb(each_channel)
def smooth_each(image, sigma):
    """
    Applies Gaussian smoothing to an RGB image, using each color channel
    independently.

    Parameters
    ----------
    image : ndarray
        Input RGB image.
    sigma : float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    result : ndarray
        Smoothed version of the input image, with each color channel processed
        independently.
    """
    return filters.gaussian(image, sigma)


@adapt_rgb(each_channel)
def mask_each(image, mask):
    """
    Applies a binary mask to an RGB image, with each color channel processed
    independently.

    Pixels where `mask` is True will be set to zero.

    Parameters
    ----------
    image : ndarray
        Input RGB image.
    mask : ndarray, bool
        Boolean mask to apply to the image.

    Returns
    -------
    result : ndarray
        Copy of the input image with masked pixels set to zero.
    """
    result = image.copy()
    result[mask] = 0
    return result


@adapt_rgb(hsv_value)
def edges_hsv(image):
    """
    Applies Sobel edge detection to an RGB image, after converting to HSV and
    using the value channel.

    Parameters
    ----------
    image : ndarray
        Input RGB image.

    Returns
    -------
    result : ndarray
        Edge map of the input image, with the value channel of the HSV image
        processed.
    """
    return filters.sobel(image)


@adapt_rgb(hsv_value)
def smooth_hsv(image, sigma):
    """
    Applies Gaussian smoothing to an RGB image, after converting to HSV and
    using the value channel.

    Parameters
    ----------
    image : ndarray
        Input RGB image.
    sigma : float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    result : ndarray
        Smoothed version of the input image, with the value channel of the HSV
        image processed.
    """
    return filters.gaussian(image, sigma)


@adapt_rgb(hsv_value)
def edges_hsv_uint(image):
    """
    Applies Sobel edge detection to an RGB image, after converting to HSV and
    using the value channel. The output is converted to unsigned 16-bit integer
    format.

    Parameters
    ----------
    image : ndarray
        Input RGB image.

    Returns
    -------
    result : ndarray, uint16
        Edge map of the input image, with the value channel of the HSV image
        processed and converted to unsigned 16-bit integer format.
    """
    return img_as_uint(filters.sobel(image))


def test_gray_scale_image():
    """Test that edges are correctly detected in a gray-scale image."""
    
    # We don't need to test both `hsv_value` and `each_channel` since
    # `adapt_rgb` is handling gray-scale inputs.
    assert_allclose(edges_each(GRAY_IMAGE), filters.sobel(GRAY_IMAGE))


def test_each_channel():
    """Test that edges are correctly detected in each color channel of an RGB image."""
        
    filtered = edges_each(COLOR_IMAGE)
    for i, channel in enumerate(np.rollaxis(filtered, axis=-1)):
        expected = img_as_float(filters.sobel(COLOR_IMAGE[:, :, i]))
        assert_allclose(channel, expected)


def test_each_channel_with_filter_argument():
    """Test that an input image with each color channel filtered with a Gaussian kernel."""
    
    filtered = smooth_each(COLOR_IMAGE, SIGMA)
    for i, channel in enumerate(np.rollaxis(filtered, axis=-1)):
        assert_allclose(channel, smooth(COLOR_IMAGE[:, :, i]))


def test_each_channel_with_asymmetric_kernel():
    """Test that an input image with each color channel masked by an upper triangular mask."""
    
    mask = np.triu(np.ones(COLOR_IMAGE.shape[:2], dtype=bool))
    mask_each(COLOR_IMAGE, mask)


def test_hsv_value():
    """Test that edges are correctly detected in the value component of the HSV representation of an RGB image."""
        
    filtered = edges_hsv(COLOR_IMAGE)
    value = color.rgb2hsv(COLOR_IMAGE)[:, :, 2]
    assert_allclose(color.rgb2hsv(filtered)[:, :, 2], filters.sobel(value))


def test_hsv_value_with_filter_argument():
    """Test that an input image with its value component filtered with a Gaussian kernel in the HSV representation."""
    
    filtered = smooth_hsv(COLOR_IMAGE, SIGMA)
    value = color.rgb2hsv(COLOR_IMAGE)[:, :, 2]
    assert_allclose(color.rgb2hsv(filtered)[:, :, 2], smooth(value))


def test_hsv_value_with_non_float_output():
    """Test that an input image with edges detected in the value component of the HSV representation,
    and the output has a dtype that matches the input dtype."""
    
    # Since `rgb2hsv` returns a float image and the result of the filtered
    # result is inserted into the HSV image, we want to make sure there isn't
    # a dtype mismatch.
    
    filtered = edges_hsv_uint(COLOR_IMAGE)
    filtered_value = color.rgb2hsv(filtered)[:, :, 2]
    value = color.rgb2hsv(COLOR_IMAGE)[:, :, 2]
    # Reduce tolerance because dtype conversion.
    assert_allclose(filtered_value, filters.sobel(value), rtol=1e-5, atol=1e-5)
