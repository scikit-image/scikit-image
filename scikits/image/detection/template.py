"""template.py - Template matching
"""
import numpy as np
import cv
import _template

def match_template_cv(image, template, out=None):
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
        Correlation results between 0.0 and 1.0. Maximum indicating most probable match.
    """
    if out == None:
        size = image.shape
        template_size = template.shape
        out = np.empty((size[0] - template_size[0] + 1,size[1] - template_size[0] + 1), dtype=np.float32)
    cv.MatchTemplate(image, template, out, cv.CV_TM_CCORR_NORMED)
    return out


def match_template(image, template):
    """Finds a template in an image using normalized correlation.

    Parameters
    ----------
    image : array_like, dtype=float
        Image to process.
    template : array_like, dtype=float
        Template to locate.
    Returns
    -------
    output : ndarray, dtype=float
        Correlation results between 0.0 and 1.0. Maximum indicating most probable match.
    """
    return _template.match_template(image, template)


if __name__ == "__main__":
    import scikits.image.io as io
    import time
    template = io.imread("../../../target.bmp").astype(np.float32)
    image = io.imread("../../../source.bmp").astype(np.float32)
    r = match_template_cv(image, template)

    t = time.time()
    result = match_template(image, template)
    print time.time() - t
    index = np.argmax(result)
    print "max sc", np.max(result), index 
    y, x = np.unravel_index(np.argmax(r), r.shape)
    print x, y
