import numpy as np
from scipy.signal import fftconvolve

from skimage.util import pad


def _window_sum(image, window_shape):

    window_sum = np.cumsum(image, axis=0)
    window_sum = (window_sum[window_shape[0]:-1]
                  - window_sum[:-window_shape[0]-1])

    window_sum = np.cumsum(window_sum, axis=1)
    window_sum = (window_sum[:, window_shape[1]:-1]
                  - window_sum[:, :-window_shape[1]-1])

    return window_sum


def match_template(image, template, pad_input=False, mode='constant',
                   constant_values=0):
    """Match a template to a 2-D image using normalized correlation.

    The output is an array with values between -1.0 and 1.0, which correspond
    to the correlation coefficient that the template is found at the position.

    Parameters
    ----------
    image : array_like
        2-D Image to process.
    template : array_like
        Template to locate.
    pad_input : bool
        If True, pad `image` with image mean so that output is the same size as
        the image, and output values correspond to the template center.
        Otherwise, the output is an array with shape `(M - m + 1, N - n + 1)`
        for an `(M, N)` image and an `(m, n)` template, and matches correspond
        to origin (top-left corner) of the template.
    mode : see `numpy.pad`, optional
        Padding mode.
    constant_values : see `numpy.pad`, optional
        Constant values used in conjunction with ``mode='constant'``.

    Returns
    -------
    output : ndarray
        Correlation results between -1.0 and 1.0. For an `(M, N)` image and an
        `(m, n)` template, the `output` is `(M - m + 1, N - n + 1)` when
        `pad_input = False` and `(M, N)` when `pad_input = True`.

    References
    ----------
    .. [1] Briechle and Hanebeck, "Template Matching using Fast Normalized
           Cross Correlation", Proceedings of the SPIE (2001).
    .. [2] J. P. Lewis, "Fast Normalized Cross-Correlation", Industrial Light
           and Magic.

    Examples
    --------
    >>> template = np.zeros((3, 3))
    >>> template[1, 1] = 1
    >>> template
    array([[ 0.  0.  0.]
           [ 0.  1.  0.]
           [ 0.  0.  0.]])
    >>> image = np.zeros((6, 6))
    >>> image[1, 1] = 1
    >>> image[4, 4] = -1
    >>> image
    array([[ 0.  0.  0.  0.  0.  0.]
           [ 0.  1.  0.  0.  0.  0.]
           [ 0.  0.  0.  0.  0.  0.]
           [ 0.  0.  0.  0.  0.  0.]
           [ 0.  0.  0.  0. -1.  0.]
           [ 0.  0.  0.  0.  0.  0.]])
    >>> result = match_template(image, template)
    >>> np.round(result, 3)
    array([[ 1.    -0.125  0.     0.   ]
           [-0.125 -0.125  0.     0.   ]
           [ 0.     0.     0.125  0.125]
           [ 0.     0.     0.125 -1.   ]])
    >>> result = match_template(image, template, pad_input=True)
    >>> np.round(result, 3)
    array([[-0.125 -0.125 -0.125  0.     0.     0.   ]
           [-0.125  1.    -0.125  0.     0.     0.   ]
           [-0.125 -0.125 -0.125  0.     0.     0.   ]
           [ 0.     0.     0.     0.125  0.125  0.125]
           [ 0.     0.     0.     0.125 -1.     0.125]
           [ 0.     0.     0.     0.125  0.125  0.125]])
    """

    if np.any(np.less(image.shape, template.shape)):
        raise ValueError("Image must be larger than template.")

    orig_shape = image.shape

    image = np.array(image, dtype=np.float32, copy=False)

    if mode == 'constant':
        image = pad(image, pad_width=template.shape, mode=mode,
                    constant_values=constant_values)
    else:
        image = pad(image, pad_width=template.shape, mode=mode)

    image_window_sum = _window_sum(image, template.shape)
    image_window_sum2 = _window_sum(image**2, template.shape)

    template_area = np.prod(template.shape)
    template_ssd = np.sum((template - template.mean())**2)

    xcorr = fftconvolve(image, template[::-1, ::-1], mode="valid")[1:-1, 1:-1]
    nom = xcorr - image_window_sum * (template.sum() / template_area)

    denom = image_window_sum2 - image_window_sum**2 / template_area
    denom *= template_ssd
    np.maximum(denom, 0, out=denom)  # sqrt of negative number not allowed
    np.sqrt(denom, out=denom)

    response = np.zeros_like(xcorr, dtype=np.float32)

    # avoid zero-division
    mask = denom > np.finfo(np.float32).eps

    response[mask] = nom[mask] / denom[mask]

    if pad_input:
        r0 = (template.shape[0] - 1) // 2
        r1 = r0 + orig_shape[0]
        c0 = (template.shape[1] - 1) // 2
        c1 = c0 + orig_shape[1]
    else:
        r0 = template.shape[0] - 1
        r1 = r0 + orig_shape[0] - template.shape[0] + 1
        c0 = template.shape[1] - 1
        c1 = c0 + orig_shape[1] - template.shape[1] + 1

    response = response[r0:r1, c0:c1]

    return response
