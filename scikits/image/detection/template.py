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
    #cv.MatchTemplate(image, template, out, cv.CV_TM_CCOEFF_NORMED)
    cv.MatchTemplate(image, template, out, cv.CV_TM_CCORR)
    return out

from scipy.signal import fftconvolve
import _template

def match_template(image, template, out=None):
    _template.match_template(image, template)

#    if out == None:
#        size = image.shape
#        template_size = template.shape
#        out = np.empty((size[0] - template_size[0] + 1,size[1] - template_size[0] + 1), dtype=np.float32)
    result = np.array(fftconvolve(image, template, mode="valid"))
    integral = np.empty((result.shape[0]+1, result.shape[1]+1))
    integral_sqr = np.empty((result.shape[0]+1, result.shape[1]+1))
    cv.Integral(result, integral, integral_sqr)
    template_sum2 = np.std(template) ** 2
    template_norm = np.mean(template) ** 2
    print integral.strides
    
#    return result

if __name__ == "__main__":
    import scikits.image.io as io
    import time
        
    target = io.imread("../../../target.bmp") #.astype(np.float32)
    image = io.imread("../../../source.bmp") #.astype(np.float32)
    t = time.time()
    result = match_template(image, target)
    print time.time() - t
