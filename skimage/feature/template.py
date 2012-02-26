"""template.py - Template matching
"""
import numpy as np
import _template

from skimage.util.dtype import _convert


def match_template(image, template, method='norm-coeff', pad_output=True):
    """Finds a template in an image using normalized correlation.

    TODO: The output is currently smaller than the input image due to
          cropping at the boundaries equal to the template width.

    Parameters
    ----------
    image : array_like
        Image to process.
    template : array_like
        Template to locate.
    method : str
        The correlation method used in scanning.
        T represents the template, I the image and R the result.
        The summation is done over X = 0..w-1 and Y = 0..h-1 of the template.
        'norm-coeff':
            R(x, y) = Sum(X,Y)[T(X, Y) * I(x + X, y + Y)] / N
            N = sqrt(Sum(X,Y)[T(X, Y)**2] * Sum(X,Y)[I(x + X, y + Y)**2])
        'norm-corr':
            R(x,y) = Sum(X,y)[T'(X, Y) * I'(x + X, y + Y)] / N
            N = sqrt(Sum(X,y)[T'(X, Y)**2] * Sum(X,Y)[I'(x + X, y + Y)**2])
            where:
            T'(x, y) = T(X, Y) - 1/(w * h) * Sum(X',Y')[T(X', Y')]
            I'(x + X, y + Y) = I(x + X, y + Y)
                               - 1/(w * h) * Sum(X',Y')[I(x + X', y + Y')]
    pad_output : bool
        If True, pad output array to be the same size as the input image.
        Otherwise, the output is an array with shape `(M - m + 1, N - n + 1)`
        for an `(M, N)` image and an `(m, n)` template.

    Returns
    -------
    output : ndarray
        Correlation results between 0.0 and 1.0, which correspond to the match
        probability when the template's *origin* (i.e. its top-left corner) is
        placed at that position. The bottom and right edges of `output` are
        truncated (`pad_output = False`) or zero-padded (`pad_output = True`),
        since otherwise the template would extend beyond the image edges.

    """
    if method not in ('norm-corr', 'norm-coeff'):
        raise ValueError("Unknown template method: %s" % method)
    image = _convert(image, np.float32)
    template = _convert(template, np.float32)
    result = _template.match_template(image, template, method)

    if pad_output:
        h, w = result.shape
        full_result = np.zeros(image.shape, dtype=np.float32)
        full_result[:h, :w] = result
        return full_result
    else:
        return result

