"""template.py - Template matching
"""
import numpy as np
import cv

def match_template_cv(image, template, out=None):
    """Calculate the absolute magnitude Sobel to find edges.

    Parameters
    ----------
    image : array_like, dtype=float
        Image to process.
    mask : array_like, dtype=bool, optional
        An optional mask to limit the application to a certain area.

    Returns
    -------
    output : ndarray
      The Sobel edge map.
    """
    if out == None:
        size = image.shape
        template_size = template.shape
        out = np.empty((size[0] - template_size[0] + 1,size[1] - template_size[0] + 1), dtype=np.float32)

    cv.MatchTemplate(image, template, out, cv.CV_TM_CCORR)
    
    index = np.argmin(out)
    print "min cv", np.min(out), index 
    cv.MatchTemplate(image, template, out, cv.CV_TM_CCORR_NORMED)
    index = np.argmax(out)
    print "max cv", np.max(out), index 
#    y, x = np.unravel_index(index, out.shape)
#    print "y x", y, x
#    print out[y, x]
#    print out[y-2:y+2, x-2:x+2]
#    result = np.array(fftconvolve(image, template, mode="valid"))
#    print "max sc", np.min(result), np.argmin(result)
#    print result[y, x]
#    print "delta", np.sum(result-out)
    return out

from scipy.signal import fftconvolve
import _template

def match_template(image, template, out=None):
    return _template.match_template(image, template)
#    if out == None:
#        size = image.shape
#        template_size = template.shape
##        out = np.empty((size[0] - template_size[0] + 1,size[1] - template_size[0] + 1), dtype=np.float32)
#    result = np.array(fftconvolve(image, template, mode="valid"))
#    integral = np.empty((result.shape[0]+1, result.shape[1]+1))
#    integral_sqr = np.empty((result.shape[0]+1, result.shape[1]+1))
#    cv.Integral(result, integral, integral_sqr)
#    template_sum2 = np.std(template) ** 2
#    template_norm = np.mean(template) ** 2
#    print integral.strides
    
#    return result

if __name__ == "__main__":
    import scikits.image.io as io
    import time
        
    template = io.imread("../../../target.bmp").astype(np.float32)
    image = io.imread("../../../source.bmp").astype(np.float32)
    r = match_template_cv(image, template)
#    print r[1, 1]
#    print np.argmax(r)
#    y, x = np.unravel_index(np.argmax(r), r.shape)
#    print x, y
#    print np.max(r)
    t = time.time()
    result = match_template(image, template)
    print time.time() - t
    index = np.argmax(result)
    print "max sc", np.max(result), index 
