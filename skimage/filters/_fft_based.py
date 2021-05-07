import numpy as np
import scipy.fft as fft


def _get_2D_butterworth_filter(nx, ny, factor, order, high_pass):
    """
    Get a 2D Butterworth mask centered on the middle of the fft

    Parameters
    ----------
    nx : int
        width of filter in pixels
    ny : int
        height of filter in pixels
    factor : float
        fraction of width and height of the filter where the cutoff should be
    order : float
        controls the slope in the cutoff region
    high_pass : bool
        whether the filter is high pass (low frequencies attenuated) or
        low pass (high frequencies are attenuated)

    Returns
    -------
    wfilt : (nx, ny) numpy array
        The filter/fft mask
    """
    lsize = np.array([nx, ny]) * factor
    # start and stop is to ensure center of mask aligns with center of fft
    gx = np.arange(-(nx - 1) // 2, (nx - 1) // 2 + 1) / lsize[0]
    gy = np.arange(-(ny - 1) // 2, (ny - 1) // 2 + 1) / lsize[1]
    qxa, qya = np.meshgrid(gx, gy)
    q2 = np.sqrt(np.square(qxa) + np.square(qya))
    wfilt = 1 / np.sqrt(1 + np.power(q2, order))
    if high_pass:
        wfilt = 1 - wfilt
    return wfilt


def butterworth(image,
                cutoff_frequency_ratio=0.005,
                high_pass=True,
                order=2.0):
    """
    This function applies a Butterworth Fourier filter to a 2D image for
    enhancing high or low frequency features.

    Parameters
    ----------
    image : numpy array
        The 2D image to be filtered
    cutoff_frequency_ratio : float, optional
        Determines the position of the cut-off relative to the
        width and height of the FFT size
    high_pass : bool, optional
        Whether to perform a high pass filter, if not a low pass filter
    order : float, optional
        Order of the filter which affects the slope near the cut-off.
        Higher order means steeper slope.

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
    if image.ndim > 2:
        raise ValueError("Only single channel images supported")
    (Ny, Nx) = image.shape
    wfilt = _get_2D_butterworth_filter(Nx, Ny, cutoff_frequency_ratio,
                                       order, high_pass)
    im_fft = fft.fftn(image)
    im_fft = fft.fftshift(im_fft)
    filtered = fft.ifftshift(wfilt * im_fft)
    butterfilt = np.real(fft.ifft2(filtered))
    return butterfilt
