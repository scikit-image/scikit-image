import numpy as np

from .shapecontext_cy import _shape_context


def shape_context(image, current_pixel, r_min=1, r_max=50, radial_bins=5,
                  polar_bins=12):
    """Compute Shape Context descriptor for a given point.

    Compute Shape Context by summing non-zero points into a log-polar
    histogram. All non-zero pixels of the image are considered to be in
    the object or shape.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image (grayscale).
    current_pixel : int tuple, (r, c)
        The pixel for which to find shape context descriptor.
    r_min : float, optional
        Minimum distance of the pixels that are considered in computation of
        histogram from `current_pixel` (default: 1).
    r_max : float, optional
        Maximum distance of the pixels that are considered in computation of
        histogram from `current_pixel` (default: 50).
    radial_bins : int, optional
        Number of log r bins in log-r vs theta histogram (default: 5).
    polar_bins : int, optional
        Number of theta bins in log-r vs theta histogram (default: 12).

    Returns
    -------
    bin_histogram : (radial_bins, polar_bins) ndarray
        The shape context - the log-r vs theta histogram of non-zero pixels in the image.

    References
    ----------
    .. [1]  Serge Belongie, Jitendra Malik and Jan Puzicha.
            "Shape matching and object recognition using shape contexts."
            IEEE PAMI 2002.
            http://www.eecs.berkeley.edu/Research/Projects/CS/vision/shape/belongie-pami02.pdf

    .. [2]  Serge Belongie, Jitendra Malik and Jan Puzicha.
            Matching with Shape Contexts
            http://www.eecs.berkeley.edu/Research/Projects/CS/vision/shape/sc_digits.html

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.feature import shape_context
    >>> img = np.zeros((20, 20)).astype(float)
    >>> img[4:8, 4:8] = 1
    >>> px = (10, 10)
    >>> shape_context(img, px, 0, 25, radial_bins=3, polar_bins=4)
    array([[  0.,   0.,   0.,   0.],
           [  0.,   0.,  16.,   0.],
           [  0.,   0.,   0.,   0.]])

    """
    # view input array as arrays with at least two dimensions
    image = np.atleast_2d(image)

    if image.ndim > 3:
        raise ValueError("Currently only supports gray-level images")

    # call helper
    bin_histogram = _shape_context(image,
                                   current_pixel[0], current_pixel[1],
                                   r_min, r_max,
                                   radial_bins, polar_bins)

    return bin_histogram
