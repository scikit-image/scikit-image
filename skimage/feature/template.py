"""template.py - Template matching
"""
import numpy as np
import _template

from skimage.util.dtype import _convert


def match_template(image, template, pad_output=True):
    """Match a template to an image using normalized correlation.

    The output is an array with values between -1.0 and 1.0, which correspond
    to the probability that the template's *origin* (i.e. its top-left
    corner) is found at that position.

    Parameters
    ----------
    image : array_like
        Image to process.
    template : array_like
        Template to locate.
    pad_output : bool
        If True, pad output array to be the same size as the input image.
        Otherwise, the output is an array with shape `(M - m + 1, N - n + 1)`
        for an `(M, N)` image and an `(m, n)` template.

    Returns
    -------
    output : ndarray
        Correlation results between -1.0 and 1.0. The `output` is truncated
        (`pad_output = False`) or zero-padded (`pad_output = True`) at the
        bottom and right edges, where the template would otherwise extend
        beyond the image edges.

    """
    image = _convert(image, np.float32)
    template = _convert(template, np.float32)
    result = _template.match_template(image, template)

    if pad_output:
        h, w = result.shape
        full_result = np.zeros(image.shape, dtype=np.float32)
        full_result[:h, :w] = result
        return full_result
    else:
        return result

