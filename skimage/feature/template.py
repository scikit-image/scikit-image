"""template.py - Template matching
"""
import numpy as np
import _template

from skimage.util.dtype import _convert


def match_template(image, template, method='norm-coeff'):
    """Finds a template in an image using normalized correlation.

    TODO: The output is currently smaller than the input image due to
          cropping at the boundaries equal to the template width.

    Parameters
    ----------
    image : array_like, dtype=float
        Image to process.
    template : array_like, dtype=float
        Template to locate.
    method: str (default 'norm-coeff')
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

    Returns
    -------
    output : ndarray, dtype=float
        Correlation results between 0.0 and 1.0, maximum indicating the most
        probable match.

    """
    if method not in ('norm-corr', 'norm-coeff'):
        raise ValueError("Unknown template method: %s" % method)
    image = _convert(image, np.float32)
    template = _convert(template, np.float32)
    return _template.match_template(image, template, method)

