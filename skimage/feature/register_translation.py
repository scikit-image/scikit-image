"""
Port of Manuel Guizar's code from:
http://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation
"""

import numpy as np


def _upsampled_dft(data, upsampled_region_size,
                   upsample_factor=1, axis_offsets=None, axes=None):
    """
    Upsampled DFT by matrix multiplication.

    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.

    Parameters
    ----------
    data : array
        The input data array (DFT of original data) to upsample.
    upsampled_region_size : integer or tuple of integers, optional
        The size of the region to be sampled.  If one integer is provided, it
        is duplicated up to the dimensionality of ``data``.
    upsample_factor : integer, optional
        The upsampling factor.  Defaults to 1.
    axis_offsets : tuple of integers, optional
        The offsets of the region to be sampled.  Defaults to None (uses
        image center)

    Returns
    -------
    output : ndarray
            The upsampled DFT of the specified region.
    """
    if axes is None:
        axes = np.arange(data.ndim)
    else:
        axes = np.array(axes)
    data_shape = np.array(data.shape)
    register_shape = data_shape[axes]
    nreg = len(axes)
    broadcast_shape = data_shape[:data.ndim - nreg]
    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = np.asarray([upsampled_region_size, ] * nreg)
    else:
        if len(upsampled_region_size) != nreg:
            raise ValueError("shape of upsampled region sizes must be equal "
                             "to input data's number of dimensions.")

    if axis_offsets is None:
        axis_offsets = np.zeros((*broadcast_shape, nreg), dtype=np.int)
    else:
        axis_offsets = np.array(axis_offsets)
        if axis_offsets.shape[-1] != nreg:
            raise ValueError("number of axis offsets must be equal to input "
                             "data's number of dimensions.")

    im2pi = 1j * 2 * np.pi

    dim_properties = list(zip(register_shape,
                         upsampled_region_size,
                         axis_offsets.T))

    for i, (n_items, ups_size, ax_offset) in enumerate(dim_properties[::-1]):
        kernel = (np.arange(ups_size)
                  * np.fft.fftfreq(n_items, upsample_factor)[:, None])
        kernel = np.exp(-im2pi * kernel)
        ax_offset = ax_offset.reshape(*broadcast_shape, 1)
        shifts = -ax_offset * np.fft.fftfreq(n_items, upsample_factor)
        shifts = np.exp(-im2pi * shifts)
        shifts = shifts.reshape(*(1, ) * i,
                                *shifts.shape[:-1],
                                *(1, ) * (nreg - i - 1),
                                shifts.shape[-1])
        data *= shifts

        s_fixed = 1
        for dim in data.shape[:-1]:
            s_fixed *= dim
        data = (data.reshape(s_fixed, data.shape[-1])
                    .dot(kernel)
                    .reshape(*data.shape[:-1], kernel.shape[-1])
                    .transpose(range(-1, data.ndim-1)))
    return_order = tuple(range(nreg, data.ndim)) + tuple(range(nreg))
    return data.transpose(return_order)


def _compute_phasediff(cross_correlation_max):
    """
    Compute global phase difference between the two images (should be
        zero if images are non-negative).

    Parameters
    ----------
    cross_correlation_max : complex
        The complex value of the cross correlation at its maximum point.
    """
    return np.arctan2(cross_correlation_max.imag, cross_correlation_max.real)


def _compute_error(cross_correlation_max, src_amp, target_amp):
    """
    Compute RMS error metric between ``src_image`` and ``target_image``.

    Parameters
    ----------
    cross_correlation_max : complex
        The complex value of the cross correlation at its maximum point.
    src_amp : float
        The normalized average image intensity of the source image
    target_amp : float
        The normalized average image intensity of the target image
    """
    error = 1.0 - cross_correlation_max * cross_correlation_max.conj() /\
        (src_amp * target_amp)
    return np.sqrt(np.abs(error))


def register_translation(src_image, target_image, upsample_factor=1,
                         space="real", return_error=True, axes=None):
    """
    Efficient subpixel image translation registration by cross-correlation.

    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.

    Parameters
    ----------
    src_image : array
        Reference image.
    target_image : array
        Image to register.  Must be same dimensionality as ``src_image``.
    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel.  Default is 1 (no upsampling)
    space : string, one of "real" or "fourier", optional
        Defines how the algorithm interprets input data.  "real" means data
        will be FFT'd to compute the correlation, while "fourier" data will
        bypass FFT of input data.  Case insensitive.
    return_error : bool, optional
        Returns error and phase difference if on,
        otherwise only shifts are returned
    axes : int or tuple of int, optional
        Register the last ``n`` axes of the images. Default None (register all
         axes)

    Returns
    -------
    shifts : ndarray
        Shift vector (in pixels) required to register ``target_image`` with
        ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)
    error : float
        Translation invariant normalized RMS error between ``src_image`` and
        ``target_image``.
    phasediff : float
        Global phase difference between the two images (should be
        zero if images are non-negative).

    References
    ----------
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008). :DOI:`10.1364/OL.33.000156`
    .. [2] James R. Fienup, "Invariant error metrics for image reconstruction"
           Optics Letters 36, 8352-8357 (1997). :DOI:`10.1364/AO.36.008352`
    """
    # images must be the same shape
    if src_image.shape != target_image.shape:
        raise ValueError("Error: images must be same size for "
                         "register_translation")

    image_dim = src_image.ndim
    image_shape = np.array(src_image.shape)

    if axes is None:
        axes = image_dim

    register_axes = np.arange(image_dim - axes, image_dim)
    register_shape = image_shape[register_axes]
    register_size = np.product(register_shape)
    broadcast_shape = image_shape[:image_dim - axes]
    register_axes = tuple(register_axes)

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        src_freq = np.fft.fftn(src_image, axes=register_axes)
        target_freq = np.fft.fftn(target_image, axes=register_axes)
    else:
        raise ValueError("Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument.")

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    cross_correlation = np.fft.ifftn(image_product, axes=register_axes)

    # Locate maximum
    flat_CC = np.abs(cross_correlation).reshape(*broadcast_shape, register_size)
    flat_maxima = np.argmax(flat_CC, axis=-1)
    maxima = np.stack(np.unravel_index(flat_maxima, register_shape), axis=-1)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in register_shape])

    shifts = maxima - np.array(register_shape) * (maxima > midpoints)
    if upsample_factor == 1:
        if return_error:
            src_amp = np.sum(np.abs(src_freq) ** 2, axis=register_axes) \
                / register_size
            target_amp = np.sum(np.abs(target_freq) ** 2, axis=register_axes) \
                / register_size
            CCmax = flat_CC[..., flat_maxima]
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5).astype(np.int)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freq.size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts*upsample_factor
        cross_correlation = _upsampled_dft(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset,
                                           axes=register_axes).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        flat_CC = np.abs(cross_correlation).reshape(*broadcast_shape,
                                                    upsampled_region_size
                                                    ** len(register_axes))
        flat_maxima = np.argmax(flat_CC, axis=-1)
        CCmax = flat_CC[..., flat_maxima]
        maxima = np.stack(np.unravel_index(flat_maxima,
                                           (upsampled_region_size,) * axes),
                          axis=-1)

        shifts = shifts + (maxima - dftshift) / upsample_factor

        if return_error:
            src_amp = _upsampled_dft(src_freq * src_freq.conj(),
                                     1, upsample_factor, axes=register_axes)
            src_amp /= normalization
            target_amp = _upsampled_dft(target_freq * target_freq.conj(),
                                        1, upsample_factor, axes=register_axes)
            target_amp /= normalization

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in register_axes:
        if shape[dim] == 1:
            shifts[dim] = 0

    if return_error:
        return shifts, _compute_error(CCmax, src_amp, target_amp),\
            _compute_phasediff(CCmax)
    else:
        return shifts
