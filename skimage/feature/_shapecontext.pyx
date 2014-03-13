#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
from six.moves import range

cimport numpy as cnp
from libc.math cimport sqrt
from libc.math cimport atan2


def _shapecontext(cnp.ndarray[cnp.float64_t, ndim=2] image,
                  float r_min, float r_max,
                  int current_pixel_x, int current_pixel_y,
                  int radial_bins=5, int polar_bins=12):
    """
    Cython implementation of calculation of shape contexts

    computes the log-polar histogram of non zero pixels with the given
    point as the origin and returns it as the shape context descriptor

    Parameters
    ----------
    image : (M, N) ndarray
        Input image (grayscale).

    r_max : float
        maximum distance of the pixels that are considered in computation
        of histogram from current_pixel

    r_min : float
        minimum distance of the pixels that are considered in computation
        of histogram from current_pixel

    current_pixel_x : int
        the row of pixel in the passed array

    current_pixel_y : int
        the column of pixel in the passed array

    radial_bins : int, optional
        number of log r bins in the log-r vs theta histogram (default: 5)

    polar_bins : int, optional
        number of theta bins in log-r vs theta histogram (default: 12)

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

    .. [2]  Wikipedia, "Shape Contexts".
            http://en.wikipedia.org/wiki/Shape_context

    """
    cdef int rows = image.shape[0]
    cdef int cols = image.shape[1]

    cdef int x, y, x_diff, y_diff, r_idx, theta_idx, tmp
    cdef float r, theta

    if r_min <= 0:
        r_min = 1

    cdef cnp.ndarray[cnp.float64_t, ndim = 1] r_array = \
        np.logspace(np.log10(r_min), np.log10(r_max), radial_bins + 1, base=10)

    cdef cnp.ndarray[cnp.float64_t, ndim = 1] theta_array = \
        np.linspace(-np.pi, np.pi, polar_bins + 1)

    cdef cnp.ndarray[cnp.float64_t, ndim = 2] bin_histogram = \
        np.zeros((radial_bins, polar_bins), dtype=float)

    cdef int r_max_int = int(r_max)

    for x in range(max(current_pixel_x - r_max_int, 0),
                   min(current_pixel_x + r_max_int, rows)):
        for y in range(max(current_pixel_y - r_max_int, 0),
                       min(current_pixel_y + r_max_int, cols)):
            if image[x, y]:  # if pixel is zero no need to consider it

                # components of position vector of the pixel from current_pixel
                x_diff = current_pixel_x - x
                y_diff = current_pixel_y - y

                # distance from current_pixel
                r = sqrt(x_diff*x_diff + y_diff*y_diff)
                # angle in radians
                theta = atan2(y_diff, x_diff)

                # find the log-r bin index of pixel in the log-polar histogram
                r_idx = -1
                for tmp in xrange(radial_bins):
                    if r > r_array[tmp] and r < r_array[tmp + 1]:
                        r_idx = tmp
                        break

                # find the polar bin index of pixel in the log-polar histogram
                theta_idx = -1
                for tmp in xrange(polar_bins):
                    if (theta > theta_array[tmp]
                            and theta < theta_array[tmp + 1]):
                        theta_idx = tmp
                        break

                # increment the counter for the point in histogram
                if r_idx != -1 and theta_idx != -1:
                    bin_histogram[r_idx, theta_idx] += 1

    return bin_histogram
