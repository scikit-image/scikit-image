import numpy as np

from ._shapecontext import _shapecontext


def shapecontext(image, r_min, r_max, current_pixel, radial_bins=5,
                 polar_bins=12):
    """Compute Shape Context descriptor for a given point.

    Compute Shape Context by summing non-zero points into a log-polar
    histogram. All non-zero pixels of the image are considered to be in
    the object or shape.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image (greyscale).

    r_max : float
        maximum distance of the pixels that are considered in computation of
        histogram from current_pixel

    r_min : float
        minmimum distance of the pixels that are considered in computation of
        histogram from current_pixel

    current_pixel : integer tuple, (r, c)
        the pixel for which to find shape context descriptor

    radial_bins : integer
        number of log r bins in log-r vs theta histogram

    polar_bins : integer
        number of theta bins in log-r vs theta histogram

    Returns
    -------
    bin_histogram : (radial_bins, polar_bins) ndarray
        the shapecontext - the log-polar histogram of points on shape

    References
    ----------
    .. [1]  Serge Belongie, Jitendra Malik and Jan Puzicha.
            "Shape matching and object recognition using shape contexts."
            IEEE PAMI 2002.
            http://www.eecs.berkeley.edu/Research/Projects/CS/vision/shape/belongie-pami02.pdf

    .. [2]  Serge Belongie, Jitendra Malik and Jan Puzicha.
            Matching with Shape Contexts
            http://www.eecs.berkeley.edu/Research/Projects/CS/vision/shape/sc_digits.html

    .. [3]  Wikipedia, "Shape Contexts".
            http://en.wikipedia.org/wiki/Shape_context

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.feature import shapecontext
    >>> img = np.zeros((20, 20)).astype(float)
    >>> img[4:8, 4:8] = 1
    >>> px = (10, 10)
    >>> shapecontext(img, 0, 25, px, radial_bins=3, polar_bins=4)
    array([[  0.,   0.,   0.,   0.],
           [  0.,   0.,  16.,   0.],
           [  0.,   0.,   0.,   0.]])

    """
    # view input array as arrays with atleast two dimensions
    image = np.atleast_2d(image)

    if image.ndim > 3:
        raise ValueError("Currently only supports grey-level images")

    # call helper
    bin_histogram = _shapecontext(image, r_min, r_max,
                                  current_pixel[0], current_pixel[1],
                                  radial_bins, polar_bins)

    return bin_histogram
