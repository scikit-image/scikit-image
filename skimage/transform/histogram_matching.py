import numpy as np


def _get_separate_channels(image):
    """Return an array consisting of all channels of an image"""
    if len(image.shape) < 3:
        return np.asarray([image])
    else:
        return image.transpose(2, 0, 1)


def _match_array_values(source, template):
    """Return modified source array with interpolated values that correspond to the
    values of the template array"""
    src_values, src_unique_indices, src_counts = \
        np.unique(source, return_inverse=True, return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template, return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts).astype(np.float64)
    src_quantiles /= src_quantiles[-1]

    tmpl_quantiles = np.cumsum(tmpl_counts).astype(np.float64)
    tmpl_quantiles /= tmpl_quantiles[-1]

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices]


def match_histograms(image, reference):
    """Adjust the pixel values of an image so that its histogram and a target's
    histogram match. The original image is preserved.

    Parameters
    ----------
    image : ndarray
        Input image. Can be gray-scale or in color
    reference : ndarray
        Image to match histogram of. Must have the same number of channels as
        image, otherwise an exception is thrown

     Returns
    -------
    matched : ndarray
        Transformed output image

    Raises
    ------
    ValueError
        Thrown when the number of channels in the input image and the reference
        differ

    References
    ----------
    .. [1] http://paulbourke.net/miscellaneous/equalisation/
    """
    shape = image.shape
    image_dtype = image.dtype

    im_channels = _get_separate_channels(image)
    ref_channels = _get_separate_channels(reference)

    if len(im_channels) != len(ref_channels):
        raise ValueError('Number of channels in the input image and reference '
                         'image must match!')

    matched_channels = np.empty(im_channels.shape)

    for channel in range(len(im_channels)):
        im_channel = im_channels[channel].ravel()
        ref_channel = ref_channels[channel].ravel()

        matched_channel = _match_array_values(im_channel, ref_channel)
        matched_channel = matched_channel.reshape(shape[0:2])
        matched_channels[channel] = matched_channel

    matched = matched_channels.transpose(1, 2, 0).reshape(shape)
    matched = np.asarray(matched, image_dtype)

    return matched
