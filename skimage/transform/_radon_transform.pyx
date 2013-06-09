#cython: cdivision=True
#cython: boundscheck=True
#cython: nonecheck=True
#cython: wraparound=False
import numpy as np
from numpy import pi

cimport numpy as cnp
cimport cython
from libc.math cimport cos, sin, floor, ceil, sqrt, abs


cpdef bilinear_ray_sum(cnp.ndarray[cnp.double_t, ndim=2] image, double theta,
                       double ray_position):
    '''Compute the projection of an image along a ray.

    Parameters
    ----------
    image : 2D array, dtype=float
        Image to project.
    :param theta: Angle of the projection.
    :param ray_position: Position of the ray within the projection

    Returns
    -------
    projected_value : float
        Ray sum along the projection
    norm_of_weights :
        A measure of how long the ray's path through the reconstruction
        circle was
    '''
    theta = theta / 180. * pi
    cdef double radius = image.shape[0] // 2 - 1
    cdef double projection_center = image.shape[0] // 2 - 1
    cdef double rotation_center = image.shape[0] // 2
    # (s, t) is the (x, y) system rotated by theta
    cdef double t = ray_position - projection_center
    # s0 is the half-length of the ray's path in the reconstruction circle
    cdef double s0
    s0 = sqrt(radius**2 - t**2) if radius**2 >= t**2 else 0.
    cdef Py_ssize_t Ns = 2 * int(ceil(2 * s0))  # number of steps along the ray
    cdef double ray_sum = 0.
    cdef double weight_norm = 0.
    cdef double ds, dx, dy, x0, y0, x, y, di, dj, index_i, index_j
    cdef Py_ssize_t k, i, j
    if Ns > 0:
        # step length between samples
        ds = 2 * s0 / Ns
        dx = ds * cos(theta)
        dy = ds * sin(theta)
        # point of entry of the ray into the reconstruction circle
        x0 = -s0 * cos(theta) + t * sin(theta)
        y0 = -s0 * sin(theta) - t * cos(theta)
        for k in range(Ns+1):
            x = x0 + k * dx
            y = y0 + k * dy
            index_i = x + rotation_center
            index_j = y + rotation_center
            i = <Py_ssize_t> floor(index_i)
            j = <Py_ssize_t> floor(index_j)
            di = index_i - floor(index_i)
            dj = index_j - floor(index_j)
            # Use linear interpolation between values
            # Where values fall outside the array, assume zero
            if i > 0 and j > 0:
                ray_sum += (1. - di) * (1. - dj) * image[i, j] * ds
                weight_norm += ((1 - di) * (1 - dj) * ds)**2
            if i > 0 and j < image.shape[1] - 1:
                ray_sum += (1. - di) * dj * image[i, j+1] * ds
                weight_norm += ((1 - di) * dj * ds)**2
            if i < image.shape[0] - 1 and j > 0:
                ray_sum += di * (1 - dj) * image[i+1, j] * ds
                weight_norm += (di * (1 - dj) * ds)**2
            if i < image.shape[0] - 1 and j < image.shape[1] - 1:
                ray_sum += di * dj * image[i+1, j+1] * ds
                weight_norm += (di * dj * ds)**2
    return ray_sum, weight_norm


cpdef bilinear_ray_update(cnp.ndarray[cnp.double_t, ndim=2] image,
        cnp.ndarray[cnp.double_t, ndim=2] image_update,
        double theta, double ray_position, double projected_value):
    """Compute the update along a ray using bilinear interpolation.

    Parameters
    ----------
    image :
        Current reconstruction estimate
    image_update :
        Array of same shape as ``image``. Updates will be added to this array.
    theta :
        Angle of the projection
    ray_position :
        Position of the ray within the projection
    projected_value :
        Projected value (from the sinogram)

    Returns
    -------
    deviation :
        Deviation before updating the image
    """
    cdef double ray_sum, weight_norm, deviation
    ray_sum, weight_norm = bilinear_ray_sum(image, theta, ray_position)
    if weight_norm > 0.:
        deviation = -(ray_sum - projected_value) / weight_norm
    else:
        deviation = 0.
    theta = theta / 180. * pi
    cdef double radius = image.shape[0] // 2 - 1
    cdef double projection_center = image.shape[0] // 2 - 1
    cdef double rotation_center = image.shape[0] // 2
    # (s, t) is the (x, y) system rotated by theta
    cdef double t = ray_position - projection_center
    # s0 is the half-length of the ray's path in the reconstruction circle
    cdef double s0
    s0 = sqrt(radius*radius - t*t) if radius**2 >= t**2 else 0.
    cdef unsigned int Ns = 2 * int(ceil(2 * s0))
    cdef double hamming_beta = 0.46164

    cdef double ds, dx, dy, x0, y0, x, y, di, dj, index_i, index_j
    cdef double hamming_window
    cdef unsigned int k, i, j
    if Ns > 0:
        # Step length between samples
        ds = 2 * s0 / Ns
        dx = ds * cos(theta)
        dy = ds * sin(theta)
        # Point of entry of the ray into the reconstruction circle
        x0 = -s0 * cos(theta) + t * sin(theta)
        y0 = -s0 * sin(theta) - t * cos(theta)
        for k in range(Ns+1):
            x = x0 + k * dx
            y = y0 + k * dy
            index_i = x + rotation_center
            index_j = y + rotation_center
            i = <Py_ssize_t> floor(index_i)
            j = <Py_ssize_t> floor(index_j)
            di = index_i - floor(index_i)
            dj = index_j - floor(index_j)
            hamming_window = ((1 - hamming_beta)
                              - hamming_beta * cos(2*pi*k / (Ns - 1)))
            if i > 0 and j > 0:
                image_update[i, j] += (deviation * (1. - di) * (1. - dj)
                                       * ds * hamming_window)
            if i > 0 and j < image.shape[1] - 1:
                image_update[i, j+1] += (deviation * (1. - di) * dj
                                         * ds * hamming_window)
            if i < image.shape[0] - 1 and j > 0:
                image_update[i+1, j] += (deviation * di * (1 - dj)
                                         * ds * hamming_window)
            if i < image.shape[0] - 1 and j < image.shape[1] - 1:
                image_update[i+1, j+1] += (deviation * di * dj
                                           * ds * hamming_window)
    return deviation


def sart_projection_update(cnp.ndarray[cnp.double_t, ndim=2] image, \
                           double theta, \
                           cnp.ndarray[cnp.double_t, ndim=1] projection,
                           double projection_shift=0.):
    cdef cnp.ndarray[cnp.double_t, ndim=2] image_update = np.zeros_like(image)
    cdef double ray_position
    cdef Py_ssize_t i
    for i in range(projection.shape[0]):
        ray_position = i + projection_shift
        bilinear_ray_update(image, image_update, theta, ray_position,
                            projection[i])
    return image_update
