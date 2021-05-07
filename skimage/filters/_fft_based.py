import numpy as np
import scipy.fft as fft
from skimage.exposure import rescale_intensity
from skimage import dtype_limits


def _get_ND_butterworth_filter(shape, factor, order, high_pass):
    """
    Get a N-dimensional Butterworth mask

    Parameters
    ----------
    shape : tuple of ints
        dimensions of the n-dimensional image and mask
    factor : float
        fraction of width and height of the filter where the cutoff should be
    order : float
        controls the slope in the cutoff region
    high_pass : bool
        whether the filter is high pass (low frequencies attenuated) or
        low pass (high frequencies are attenuated)

    Returns
    -------
    wfilt : numpy array of shape
        The filter/fft mask
    """
    ranges = []
    for d in shape:
        # start and stop ensures center of mask aligns with center of fft
        axis = np.arange(-(d - 1) // 2, (d - 1) // 2 + 1) / (d * factor)
        ranges.append(axis**2)
    # q2 = euclidian distance squared grid
    q2 = np.sum(np.meshgrid(*ranges, indexing="ij"), axis=0)
    # division of order by 2 to avoid additional square root on q2
    wfilt = 1 / np.sqrt(1 + np.power(q2, order/2))
    if high_pass:
        wfilt = 1 - wfilt
    return fft.ifftshift(wfilt)


def butterworth(image,
                cutoff_frequency_ratio=0.005,
                high_pass=True,
                order=2.0,
                channel_axis=None,
                preserve_range=False,
                ):
    """
    This function applies a Butterworth Fourier filter to an N dimensional
    image for enhancing high or low frequency features.

    Parameters
    ----------
    image : numpy array
        The 2D image to be filtered
    cutoff_frequency_ratio : float, optional
        Determines the position of the cut-off relative to the
        shape of the FFT
    high_pass : bool, optional
        Whether to perform a high pass filter. If False, a low pass filter is
        performed.
    order : float, optional
        Order of the filter which affects the slope near the cut-off.
        Higher order means steeper slope in frequency space.
    channel_axis : int, optional
        If there is a channel dimension, provide the index here. If None
        (default) then all axes are assumed spatial dimensions.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise the input
        image is returned with dtype float with values scaled between 0 and 1.

    Returns
    -------
    result : numpy array
        The Butterworth-filtered image

    Notes
    -----
    * A band-pass filter can be achieved by combining a high pass and low
    pass filter
    * cutoff_frequency_ratio and order both affect slope at the cut-off
    region. If a specific slope is desired in this region, its absolute value
    is approximately equal to $order / factor * 2^(-2.5)$

    Reference
    --------
    Butterworth, Stephen. "On the theory of filter amplifiers."
    Wireless Engineer 7.6 (1930): 536-541.
    """
    fft_shape = (image.shape if channel_axis is None
                 else np.delete(image.shape, channel_axis))
    wfilt = _get_ND_butterworth_filter(fft_shape, cutoff_frequency_ratio,
                                       order, high_pass)
    butterfilt = np.empty(image.shape, dtype=np.float64)
    if channel_axis is None:
        im_fft = fft.fftn(image)
        butterfilt[:] = np.real(fft.ifftn(wfilt * im_fft))
    else:
        abs_channel = channel_axis % image.ndim
        post = image.ndim - abs_channel - 1
        for i in range(image.shape[channel_axis]):
            sl = (slice(None),)*abs_channel + (i,) + (slice(None),)*post
            im_fft = fft.fftn(image[sl])
            butterfilt[sl] = np.real(fft.ifftn(wfilt * im_fft))
    out_range = dtype_limits(image) if preserve_range else (0., 1.)
    butterfilt = rescale_intensity(butterfilt, out_range=out_range)
    return butterfilt
