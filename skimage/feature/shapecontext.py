import numpy as np

from ._shapecontext import _shapecontext

def shapecontext(image, r_min, r_max, current_pixel, radial_bins=5, polar_bins=12):
    """Compute Shape Context descriptor for a given point.

    Compute Shape Context by summing non-zero points into a log-polar histogram.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image (greyscale).

    r_max : 
        maximum distance of the pixels that are considered in computation of histogram
        from current_pixel

    r_min : 
        minmimum distance of the pixels that are considered in computation of histogram
        from current_pixel

    current_pixel : (r, c) tuple
        the pixel for which to find shape context descriptor

    radial_bins : number of log r bins in log-r vs theta histogram

    polar_bins : number of theta bins

    Returns
    -------
    the shapecontext - the log-polar histogram of points relative to other points on the shape

    References
    ----------
    * http://en.wikipedia.org/wiki/Shape_context

    * Serge Belongie, Jitendra Malik and Jan Puzicha. "Shape matching and 
    object recognition using shape contexts." Pattern Analysis and Machine
    Intelligence, IEEE Transactions on. 24.4 (2002): 509-522.

    Usage
    -----



    """
    image = np.atleast_2d(image)

    if image.ndim > 3:
        raise ValueError("Currently only supports grey-level images")    

    bin_histogram = _shapecontext(image, r_min, r_max,
                                  current_pixel[0], current_pixel[1],
                                  radial_bins, polar_bins)

    return bin_histogram
