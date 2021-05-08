import numpy as np
import functools
from .._shared.fft import fftmodule as fft
from ..exposure import rescale_intensity
from ..util import dtype_limits


def _get_ND_butterworth_filter(shape, factor, order, high_pass, real):
    """
    Create a N-dimensional Butterworth mask for an FFT

    Parameters
    ----------
    shape : tuple of int
        Shape of the n-dimensional FFT and mask.
    factor : float
        Fraction of mask dimensions where the cutoff should be.
    order : float
        Controls the slope in the cutoff region.
    high_pass : bool
        Whether the filter is high pass (low frequencies attenuated) or
        low pass (high frequencies are attenuated).
    real : bool
        Whether the FFT is of a real (True) or complex (False) image

    Returns
    -------
    wfilt : ndarray
        The FFT mask.
    """
    ranges = []
    for i, d in enumerate(shape):
        # start and stop ensures center of mask aligns with center of FFT
        axis = np.arange(-(d - 1) // 2, (d - 1) // 2 + 1) / (d * factor)
        ranges.append(fft.ifftshift(axis ** 2))
    # for real image FFT, halve the last axis
    if real:
        limit = d // 2 + 1
        ranges[-1] = ranges[-1][:limit]
    # q2 = squared Euclidian distance grid
    q2 = functools.reduce(
            np.add, np.meshgrid(*ranges, indexing="ij", sparse=True)
            )
    # division of order by 2 to avoid additional square root on q2
    wfilt = 1 / np.sqrt(1 + np.power(q2, order / 2))
    if high_pass:
        wfilt = 1 - wfilt
    return wfilt


def butterworth(
    image,
    cutoff_frequency_ratio=0.005,
    high_pass=True,
    order=2.0,
    channel_axis=None,
    preserve_range=False,
):
    """Apply a Butterworth filter to enhance high or low frequency features.

    This filter is defined in the Fourier domain

    Parameters
    ----------
    image : (M[, N[, ..., P]][, C]) ndarray
        Input image.
    cutoff_frequency_ratio : float, optional
        Determines the position of the cut-off relative to the
        shape of the FFT.
    high_pass : bool, optional
        Whether to perform a high pass filter. If False, a low pass filter is
        performed.
    order : float, optional
        Order of the filter which affects the slope near the cut-off.
        Higher order means steeper slope in frequency space.
    channel_axis : int, optional
        If there is a channel dimension, provide the index here. If None
        (default) then all axes are assumed to be spatial dimensions.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise the output
        image is returned with dtype float with values scaled between 0 and 1.
        If a complex valued image was passed, no rescaling is performed.

    Returns
    -------
    result : ndarray
        The Butterworth-filtered image.

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
    is_real = np.isrealobj(image)
    wfilt = _get_ND_butterworth_filter(
        fft_shape, cutoff_frequency_ratio, order, high_pass, is_real
    )
    if is_real:
        fftf = fft.rfftn
        ifftf = fft.irfftn
    else:
        fftf = fft.fftn
        ifftf = fft.ifftn
    axes = np.arange(image.ndim)
    if channel_axis is not None:
        abs_channel = channel_axis % image.ndim
        post = image.ndim - abs_channel - 1
        sl = ((slice(None),) * abs_channel + (np.newaxis,) +
              (slice(None),) * post)
        axes = np.delete(axes, channel_axis)
        wfilt = wfilt[sl]
    butterfilt = ifftf(wfilt * fftf(image, axes=axes), s=fft_shape, axes=axes)
    if is_real:
        out_range = dtype_limits(image) if preserve_range else (0.0, 1.0)
        butterfilt = rescale_intensity(butterfilt, out_range=out_range)
    return butterfilt
