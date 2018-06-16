import numpy as np


def _get_separate_channels(image):
    """Return an array consisting of all channels of an image"""
    if len(image.shape) < 3:
        return np.asarray([image])
    else:
        return image.transpose(2, 0, 1)


def _match_array_values(a, b):
    """Return modified a array with interpolated values that correspond to the values of the b array"""
    a_values, a_unique_indices, a_counts = np.unique(a, return_inverse=True, return_counts=True)
    b_values, b_counts = np.unique(b, return_counts=True)

    # calculate normalized quantiles for each array
    a_quantiles = np.cumsum(a_counts).astype(np.float64)
    a_quantiles /= a_quantiles[-1]

    b_quantiles = np.cumsum(b_counts).astype(np.float64)
    b_quantiles /= b_quantiles[-1]

    interp_a_values = np.interp(a_quantiles, b_quantiles, b_values)
    return interp_a_values[a_unique_indices]


def match_histograms(image, reference):
    """Adjust the pixel values of an image so that its histogram and a target's histogram match. The original image
    is preserved

    Parameters
    ----------
    image : ndarray
        Input image. Can be gray-scale or in color
    reference : ndarray
        Image to match histogram of.

     Returns
    -------
    matched : ndarray
        Transformed output image.

    References
    ----------
    .. [1] http://paulbourke.net/miscellaneous/equalisation/
    """
    shape = image.shape

    im_channels = _get_separate_channels(image)
    ref_channels = _get_separate_channels(reference)

    if len(im_channels) != len(ref_channels):
        raise ValueError('Number of channels in the input image and reference image must match!')

    matched_channels = np.empty(im_channels.shape)

    for channel in range(len(im_channels)):
        im_channel = im_channels[channel].ravel()
        ref_channel = ref_channels[channel].ravel()

        matched_channel = _match_array_values(im_channel, ref_channel)
        matched_channel = matched_channel.reshape(shape[0:2])
        matched_channels[channel] = matched_channel

    matched = matched_channels.transpose(1, 2, 0).reshape(shape)

    return matched
