import numpy as np
from scipy import sqrt, pi, arctan2, cos, sin

from ._shapecontext import _shapecontext

def shapecontext(image, r_min, r_max, current_pixel, radial_bins=5, polar_bins=12):
    """Compute Shape Context descriptor for a given point.

    Compute Shape Context by


    Parameters
    ----------
    image : (M, N) ndarray
        Input image (greyscale).

    Returns
    -------

    References
    ----------
    * http://en.wikipedia.org/wiki/Shape_context

    * S. Belongie and J. Malik. Matching with Shape Contexts 

    """
    image = np.atleast_2d(image)

    if image.ndim > 3:
        raise ValueError("Currently only supports grey-level images")    

    bin_histogram = _shapecontext(image, r_min, r_max,
                                  current_pixel[0], current_pixel[1],
                                  radial_bins, polar_bins)

    return bin_histogram
