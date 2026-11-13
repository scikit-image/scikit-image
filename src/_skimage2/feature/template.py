import math

import numpy as np
from scipy import fft

from _skimage2._shared.utils import check_nD, _supported_float_type


def _window_sum_2d(image, window_shape):
    window_sum = np.cumsum(image, axis=0)
    window_sum = window_sum[window_shape[0] : -1] - window_sum[: -window_shape[0] - 1]

    window_sum = np.cumsum(window_sum, axis=1)
    window_sum = (
        window_sum[:, window_shape[1] : -1] - window_sum[:, : -window_shape[1] - 1]
    )

    return window_sum


def _window_sum_3d(image, window_shape):
    window_sum = _window_sum_2d(image, window_shape)

    window_sum = np.cumsum(window_sum, axis=2)
    window_sum = (
        window_sum[:, :, window_shape[2] : -1]
        - window_sum[:, :, : -window_shape[2] - 1]
    )

    return window_sum


def _valid_xcorr(image, template):
    """Compute only fully overlapping cross-correlation windows.

    A circular convolution with transform lengths at least ``image.shape``
    agrees with the linear convolution on indices ``template.shape - 1``
    through ``image.shape - 1``. Wrapped tails only enter the discarded prefix
    on at least one axis. Avoiding the full linear-convolution padding saves
    substantial memory for large templates.
    """
    complex_result = np.iscomplexobj(template)
    transform = fft.fftn if complex_result else fft.rfftn
    inverse = fft.ifftn if complex_result else fft.irfftn
    fshape = tuple(fft.next_fast_len(n, real=not complex_result) for n in image.shape)
    image_fft = transform(image, fshape)
    reverse = (slice(None, None, -1),) * template.ndim
    template_fft = transform(template[reverse], fshape)
    # Reuse the wider transform, preserving mixed-dtype promotion.
    if image_fft.dtype.itemsize >= template_fft.dtype.itemsize:
        image_fft *= template_fft
        del template_fft
        spectrum = image_fft
    else:
        template_fft *= image_fft
        del image_fft
        spectrum = template_fft
    result = inverse(spectrum, fshape, overwrite_x=True)
    valid = tuple(
        slice(width, size - 1) for width, size in zip(template.shape, image.shape)
    )
    # The one-sample outer border is used only by the window-sum helpers.
    return result[valid].copy()


def match_template(
    image, template, pad_input=False, mode='constant', constant_values=0
):
    """Match a template to a 2-D or 3-D image using normalized correlation.

    The output is an array with values between -1.0 and 1.0. The value at a
    given position corresponds to the correlation coefficient between the image
    and the template.

    For `pad_input=True` matches correspond to the center and otherwise to the
    top-left corner of the template. To find the best match you must search for
    peaks in the response (output) image.

    Parameters
    ----------
    image : (M, N[, P]) array
        2-D or 3-D input image.
    template : (m, n[, p]) array
        Template to locate. It must be `(m <= M, n <= N[, p <= P])`.
    pad_input : bool
        If True, pad `image` so that output is the same size as the image, and
        output values correspond to the template center. Otherwise, the output
        is an array with shape `(M - m + 1, N - n + 1)` for an `(M, N)` image
        and an `(m, n)` template, and matches correspond to origin
        (top-left corner) of the template.
    mode : see `numpy.pad`, optional
        Padding mode.
    constant_values : see `numpy.pad`, optional
        Constant values used in conjunction with ``mode='constant'``.

    Returns
    -------
    output : array
        Response image with correlation coefficients.

    Notes
    -----
    Details on the cross-correlation are presented in [1]_. This implementation
    uses FFT convolutions of the image and the template. Reference [2]_
    presents similar derivations but the approximation presented in this
    reference is not used in our implementation.

    References
    ----------
    .. [1] J. P. Lewis, "Fast Normalized Cross-Correlation", Industrial Light
           and Magic.
    .. [2] Briechle and Hanebeck, "Template Matching using Fast Normalized
           Cross Correlation", Proceedings of the SPIE (2001).
           :DOI:`10.1117/12.421129`

    Examples
    --------
    >>> template = np.zeros((3, 3))
    >>> template[1, 1] = 1
    >>> template
    array([[0., 0., 0.],
           [0., 1., 0.],
           [0., 0., 0.]])
    >>> image = np.zeros((6, 6))
    >>> image[1, 1] = 1
    >>> image[4, 4] = -1
    >>> image
    array([[ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0., -1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.]])
    >>> result = match_template(image, template)
    >>> np.round(result, 3)
    array([[ 1.   , -0.125,  0.   ,  0.   ],
           [-0.125, -0.125,  0.   ,  0.   ],
           [ 0.   ,  0.   ,  0.125,  0.125],
           [ 0.   ,  0.   ,  0.125, -1.   ]])
    >>> result = match_template(image, template, pad_input=True)
    >>> np.round(result, 3)
    array([[-0.125, -0.125, -0.125,  0.   ,  0.   ,  0.   ],
           [-0.125,  1.   , -0.125,  0.   ,  0.   ,  0.   ],
           [-0.125, -0.125, -0.125,  0.   ,  0.   ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.125,  0.125,  0.125],
           [ 0.   ,  0.   ,  0.   ,  0.125, -1.   ,  0.125],
           [ 0.   ,  0.   ,  0.   ,  0.125,  0.125,  0.125]])
    """
    check_nD(image, (2, 3))

    if image.ndim < template.ndim:
        raise ValueError(
            "Dimensionality of template must be less than or "
            "equal to the dimensionality of image."
        )
    if np.any(np.less(image.shape, template.shape)):
        raise ValueError("Image must be larger than template.")

    image_shape = image.shape

    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    # The integral-image helpers and the cropped valid convolution need one
    # border sample on each side. Extra padding is needed only for responses
    # whose template overlaps the image boundary.
    if pad_input:
        pad_width = tuple(
            (width // 2 + 1, (width - 1) // 2 + 1) for width in template.shape
        )
    else:
        pad_width = ((1, 1),) * image.ndim
    if mode == 'constant':
        image = np.pad(
            image, pad_width=pad_width, mode=mode, constant_values=constant_values
        )
    else:
        # Some modes (e.g. linear_ramp and user callbacks) depend on the full
        # pad width. Preserve that operation, then discard the unused border.
        full_pad_width = tuple((width, width) for width in template.shape)
        image = np.pad(image, pad_width=full_pad_width, mode=mode)
        crop = tuple(
            slice(width - before, width + size + after)
            for width, size, (before, after) in zip(
                template.shape, image_shape, pad_width
            )
        )
        image = image[crop].copy()

    template_mean = template.mean()
    template_volume = math.prod(template.shape)
    template_ssd = np.sum((template - template_mean) ** 2)

    # Compute the FFT before allocating the window sums: its work arrays can
    # be considerably larger than the input. Copy the result to release any
    # larger convolution array retained by the cropped view.
    xcorr = _valid_xcorr(image, template)

    # Use special case for 2-D images for much better performance in
    # computation of integral images.
    window_sum = _window_sum_2d if image.ndim == 2 else _window_sum_3d
    image_window_sum = window_sum(image, template.shape)
    xcorr -= image_window_sum * template_mean

    # np.pad owns this array, so it is safe to reuse it for squared values.
    np.square(image, out=image)
    image_window_sum2 = window_sum(image, template.shape)
    del image

    denominator = image_window_sum2
    np.multiply(image_window_sum, image_window_sum, out=image_window_sum)
    np.divide(image_window_sum, template_volume, out=image_window_sum)
    denominator -= image_window_sum
    del image_window_sum
    denominator *= template_ssd
    np.maximum(denominator, 0, out=denominator)  # sqrt of negative number not allowed
    np.sqrt(denominator, out=denominator)

    response = np.zeros_like(xcorr, dtype=float_dtype)

    # avoid zero-division
    mask = denominator > np.finfo(float_dtype).eps

    np.divide(xcorr, denominator, out=response, where=mask)
    return response
