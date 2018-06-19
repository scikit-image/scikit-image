import numpy as np


def _match_cumulative_cdf(source, template):
    """Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template"""
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
    """Adjust an image so that its cumulative histogram matches that of another.

    The adjustment is applied separately for each channel.

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

    if image.ndim > 2:
        image = image.transpose(2, 0, 1)
    image = np.array(image, copy=False, ndmin=3)

    if reference.ndim > 2:
        reference = reference.transpose(2, 0, 1)
    reference = np.array(reference, copy=False, ndmin=3)

    if len(image) != len(reference):
        raise ValueError('Number of channels in the input image and reference '
                         'image must match!')

    matched_channels = np.empty(image.shape)

    for channel in range(len(image)):
        im_channel = image[channel].ravel()
        ref_channel = reference[channel].ravel()

        matched_channel = _match_cumulative_cdf(im_channel, ref_channel)
        matched_channel = matched_channel.reshape(shape[0:2])
        matched_channels[channel] = matched_channel

    matched = matched_channels.transpose([1, 2, 0]).reshape(shape)
    matched = np.asarray(matched, image_dtype)

    return matched
