"""template.py - Template matching
"""
import numpy as np
from . import _template


def match_template(image, template, pad_input=False):
    """Match a template to an image using normalized correlation.

    The output is an array with values between -1.0 and 1.0, which correspond
    to the probability that the template is found at that position.

    Parameters
    ----------
    image : array_like
        Image to process.
    template : array_like
        Template to locate.
    pad_input : bool
        If True, pad `image` with image mean so that output is the same size as
        the image, and output values correspond to the template center.
        Otherwise, the output is an array with shape `(M - m + 1, N - n + 1)`
        for an `(M, N)` image and an `(m, n)` template, and matches correspond
        to origin (top-left corner) of the template.

    Returns
    -------
    output : ndarray
        Correlation results between -1.0 and 1.0. For an `(M, N)` image and an
        `(m, n)` template, the `output` is `(M - m + 1, N - n + 1)` when
        `pad_input = False` and `(M, N)` when `pad_input = True`.

    Examples
    --------
    >>> template = np.zeros((3, 3))
    >>> template[1, 1] = 1
    >>> print template
    [[ 0.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  0.]]
    >>> image = np.zeros((6, 6))
    >>> image[1, 1] = 1
    >>> image[4, 4] = -1
    >>> print image
    [[ 0.  0.  0.  0.  0.  0.]
     [ 0.  1.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0. -1.  0.]
     [ 0.  0.  0.  0.  0.  0.]]
    >>> result = match_template(image, template)
    >>> print np.round(result, 3)
    [[ 1.    -0.125  0.     0.   ]
     [-0.125 -0.125  0.     0.   ]
     [ 0.     0.     0.125  0.125]
     [ 0.     0.     0.125 -1.   ]]
    >>> result = match_template(image, template, pad_input=True)
    >>> print np.round(result, 3)
    [[-0.125 -0.125 -0.125  0.     0.     0.   ]
     [-0.125  1.    -0.125  0.     0.     0.   ]
     [-0.125 -0.125 -0.125  0.     0.     0.   ]
     [ 0.     0.     0.     0.125  0.125  0.125]
     [ 0.     0.     0.     0.125 -1.     0.125]
     [ 0.     0.     0.     0.125  0.125  0.125]]
    """
    if np.any(np.less(image.shape, template.shape)):
        raise ValueError("Image must be larger than template.")
    image = np.ascontiguousarray(image, dtype=np.float32)
    template = np.ascontiguousarray(template, dtype=np.float32)

    if pad_input:
        pad_size = tuple(np.array(image.shape) + np.array(template.shape) - 1)
        pad_image = np.mean(image) * np.ones(pad_size, dtype=np.float32)
        h, w = image.shape
        i0, j0 = template.shape
        i0 /= 2
        j0 /= 2
        pad_image[i0:i0 + h, j0:j0 + w] = image
        image = pad_image
    result = _template.match_template(image, template)
    return result
