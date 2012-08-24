'''gauss.py - Gaussian image filter

Reference: http://en.wikipedia.org/wiki/Gaussian_filter

Author: Riaan van den Dool

'''

import numpy as np
from .lpi_filter import LPIFilter2D 
from scipy import exp

def gaussian(image, sigma=1.):
    '''Gaussian filter (blur) an image.

    Parameters
    -----------
    image : array_like, dtype=float
      The greyscale input image to filter/blur

    sigma : float or list/tuple 
      The standard deviation of the Gaussian filter
      If float is passed, the same sigma is applied in both dimensions,
      if a tuple or list is passed, the first element is used as sigma
      for dimension 0 (y) and the second element is used as sigma for
      dimension 1 (x).

    Returns
    -------
    output : array (image), dtype=float
      The gaussian blurred image.

    References
    -----------
    http://en.wikipedia.org/wiki/Gaussian_filter

    Examples
    --------
    >>> from skimage import filter
    >>> # Generate noisy image of a square
    >>> im = np.zeros((256, 256))
    >>> im[64:-64, 64:-64] = 1
    >>> im += 0.2*np.random.random(im.shape)
    >>> # Applying a simple gaussian filter
    >>> blurred = filter.gaussian(im, sigma=0.8)
    >>> # Applying an elliptical gauss filter
    >>> blurred = filter.gaussian(im, sigma=(0.8, 0.5))
    '''

    if image.ndim != 2:
        raise TypeError("The input 'image' must be a two dimensional array.")

    if isinstance(sigma, (list, tuple)):
        sigmax = float(sigma[1])
        sigmay = float(sigma[0])
    else:
        sigmax = float(sigma)
        sigmay = float(sigma)
    
    def filt_func(r, c):
        return exp(-(c**2/(2*sigmax**2)+r**2/(2*sigmay**2)))

    gauss_filter = LPIFilter2D(filt_func)

    return gauss_filter(image)
