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
        Correlation results between 0.0 and 1.0, maximum indicating the most probable match.
    """
    if out == None:
        out = np.empty((image.shape[0] - template.shape[0] + 1,image.shape[1] - template.shape[1] + 1), dtype=image.dtype)
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
        Correlation results between 0.0 and 1.0, maximum indicating the most probable match.
    """
    return _template.match_template(image, template)


if __name__ == "__main__":
    import scikits.image.io as io
    import time
    template = io.imread("../../../target.bmp").astype(np.float32)
    temp2 = np.empty((template.shape[0] + 80, template.shape[1]), dtype=template.dtype)
    cv.Resize(np.ascontiguousarray(template), temp2)
    template = temp2
    image = io.imread("../../../source.bmp").astype(np.float32)
    
    t = time.time()
    r = match_template_cv(image, template)
    print "cv", time.time() - t
    index = np.argmax(r)
    print "max cv", np.max(r), index 
    y, x = np.unravel_index(index, r.shape)
    print x, y
    
    
    print template.strides
    print template.shape
    t = time.time()
    result = match_template(image, template)
    print "sc", time.time() - t
    
    index = np.argmax(result)
    print "max sc", np.max(result), index 
    y2, x2 = np.unravel_index(index, result.shape)
    print x2, y2
    
    io.use_plugin("gtk")
    
    output = image.astype(np.uint8)
    cv.Rectangle(output, (x, y), (x + template.shape[1],y + template.shape[0]), (0,0,255))
    cv.Rectangle(output, (x2, y2), (x2 + template.shape[1],y2 + template.shape[0]), (0,0,255))
    io.imshow(output)
    io.show()
