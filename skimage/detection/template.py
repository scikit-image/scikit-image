"""template.py - Template matching
"""
import numpy as np
import cv
import _template

#XXX add to opencv backend once backend system in place
def match_template_cv(image, template, out=None,  method="norm-coeff"):
    """Finds a template in an image using normalized correlation.

    Parameters
    ----------
    image : array_like, dtype=float
        Image to process.
    template : array_like, dtype=float
        Template to locate.
    out: array_like, dtype=float, optional
        Optional destination.
    Returns
    -------
    output : ndarray, dtype=float
        Correlation results between 0.0 and 1.0, maximum indicating the most probable match.
    """
    if out == None:
        out = np.empty((image.shape[0] - template.shape[0] + 1,image.shape[1] - template.shape[1] + 1), dtype=image.dtype)
    if method == "norm-corr":
        cv.MatchTemplate(image, template, out, cv.CV_TM_CCORR_NORMED)
    elif method == "norm-corr":
        cv.MatchTemplate(image, template, out, cv.CV_TM_CCOEFF_NORMED)
    else:
        raise ValueError("Unknown template method: %s" % method)
    return out


def match_template(image, template, method="norm-coeff"):
    """Finds a template in an image using normalized correlation.

    Parameters
    ----------
    image : array_like, dtype=float
        Image to process.
    template : array_like, dtype=float
        Template to locate.
    method: str (default 'norm-coeff')
        The correlation method used in scanning.
        T represents the template, I the image and R the result.        
        The summation is done over x' = 0..w-1 and y' = 0..h-1 of the template.
        'norm-coeff': 
            R(x, y) = Sigma(x',y')[T(x', y').I(x + x', y + y')] / N
            N = sqrt(Sigma(x',y')[T(x', y')**2].Sigma(x',y')[I(x + x', y + y')**2])
        'norm-corr': 
            R(x,y) = Sigma(x',y)[T'(x', y').I'(x + x', y + y')] / N 
            N = sqrt(Sigma(x',y)[T'(x', y')**2].Sigma(x',y')[I'(x + x', y + y')**2])
            where:
            T'(x, y) = T(x', y') - 1/(w.h).Sigma(x'',y'')[T(x'', y'')]
            I'(x + x', y + y') =  I(x + x', y + y') - 
                1/(w.h).Sigma(x'',y'')[I(x + x'', y + y'')]

    Returns
    -------
    output : ndarray, dtype=float
        Correlation results between 0.0 and 1.0, maximum indicating the most
        probable match.
    """
    if method == "norm-corr":
        method_num = 0
    elif method == "norm-coeff":
        method_num = 1
    else:
        raise ValueError("Unknown template method: %s" % method)
    return _template.match_template(image, template, method_num)


    

