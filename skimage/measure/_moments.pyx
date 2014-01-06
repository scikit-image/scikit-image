#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np


def moments(double[:, :] image, Py_ssize_t order=3):
    """Calculate all raw image moments up to a certain order.

    The following properties can be calculated from raw image moments:
     * Area as ``m[0, 0]``.
     * Centroid as {``m[0, 1] / m[0, 0]``, ``m[1, 0] / m[0, 0]``}.

    Note that raw moments are neither translation, scale nor rotation
    invariant.

    Parameters
    ----------
    image : 2D double array
        Rasterized shape as image.
    order : int, optional
        Maximum order of moments. Default is 3.

    Returns
    -------
    m : (``order + 1``, ``order + 1``) array
        Raw image moments.

    References
    ----------
    .. [1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [2] B. Jähne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [4] http://en.wikipedia.org/wiki/Image_moment

    """
    return moments_central(image, 0, 0, order)


def moments_central(double[:, :] image, double cr, double cc,
                    Py_ssize_t order=3):
    """Calculate all central image moments up to a certain order.

    Note that central moments are translation invariant but not scale and
    rotation invariant.

    Parameters
    ----------
    image : 2D double array
        Rasterized shape as image.
    cr : double
        Center row coordinate.
    cc : double
        Center column coordinate.
    order : int, optional
        Maximum order of moments. Default is 3.

    Returns
    -------
    mu : (``order + 1``, ``order + 1``) array
        Central image moments.

    References
    ----------
    .. [1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [2] B. Jähne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [4] http://en.wikipedia.org/wiki/Image_moment

    """
    cdef Py_ssize_t p, q, r, c
    cdef double[:, ::1] mu = np.zeros((order + 1, order + 1), dtype=np.double)
    for p in range(order + 1):
        for q in range(order + 1):
            for r in range(image.shape[0]):
                for c in range(image.shape[1]):
                    mu[p, q] += image[r, c] * (r - cr) ** q * (c - cc) ** p
    return np.asarray(mu)


def moments_normalized(double[:, :] mu, Py_ssize_t order=3):
    """Calculate all normalized central image moments up to a certain order.

    Note that normalized central moments are translation and scale invariant
    but not rotation invariant.

    Parameters
    ----------
    mu : (M, M) array
        Central image moments, where M must be > ``order``.
    order : int, optional
        Maximum order of moments. Default is 3.

    Returns
    -------
    nu : (``order + 1``, ``order + 1``) array
        Normalized central image moments.

    References
    ----------
    .. [1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [2] B. Jähne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [4] http://en.wikipedia.org/wiki/Image_moment

    """
    cdef Py_ssize_t p, q
    cdef double[:, ::1] nu = np.zeros((order + 1, order + 1), dtype=np.double)
    for p in range(order + 1):
        for q in range(order + 1):
            if p + q >= 2:
                nu[p, q] = mu[p, q] / mu[0, 0] ** (<double>(p + q) / 2 + 1)
            else:
                nu[p, q] = np.nan
    return np.asarray(nu)


def moments_hu(double[:, :] nu):
    """Calculate Hu's set of image moments.

    Note that this set of moments is proofed to be translation, scale and
    rotation invariant.

    Parameters
    ----------
    nu : (M, M) array
        Normalized central image moments, where M must be > 4.

    Returns
    -------
    nu : (7, 1) array
        Hu's set of image moments.

    References
    ----------
    .. [1] M. K. Hu, "Visual Pattern Recognition by Moment Invariants",
           IRE Trans. Info. Theory, vol. IT-8, pp. 179-187, 1962
    .. [2] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [3] B. Jähne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [4] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [5] http://en.wikipedia.org/wiki/Image_moment


    """
    cdef double[::1] hu = np.zeros((7, ), dtype=np.double)
    cdef double t0 = nu[3, 0] + nu[1, 2]
    cdef double t1 = nu[2, 1] + nu[0, 3]
    cdef double q0 = t0 * t0
    cdef double q1 = t1 * t1
    cdef double n4 = 4 * nu[1, 1]
    cdef double s = nu[2, 0] + nu[0, 2]
    cdef double d = nu[2, 0] - nu[0, 2]
    hu[0] = s
    hu[1] = d * d + n4 * nu[1, 1]
    hu[3] = q0 + q1
    hu[5] = d * (q0 - q1) + n4 * t0 * t1
    t0 *= q0 - 3 * q1
    t1 *= 3 * q0 - q1
    q0 = nu[3, 0]- 3 * nu[1, 2]
    q1 = 3 * nu[2, 1] - nu[0, 3]
    hu[2] = q0 * q0 + q1 * q1
    hu[4] = q0 * t0 + q1 * t1
    hu[6] = q1 * t0 - q0 * t1
    return np.asarray(hu)
