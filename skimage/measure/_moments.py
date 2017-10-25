# coding: utf-8
import numpy as np
from . import _moments_cy
import itertools
from warnings import warn


def moments(image, order=3):
    """Calculate all raw image moments up to a certain order.

    The following properties can be calculated from raw image moments:
     * Area as: ``M[0, 0]``.
     * Centroid as: {``M[1, 0] / M[0, 0]``, ``M[0, 1] / M[0, 0]``}.

    Note that raw moments are neither translation, scale nor rotation
    invariant.

    Parameters
    ----------
    image : 2D double or uint8 array
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

    Examples
    --------
    >>> image = np.zeros((20, 20), dtype=np.double)
    >>> image[13:17, 13:17] = 1
    >>> M = moments(image)
    >>> cr = M[1, 0] / M[0, 0]
    >>> cc = M[0, 1] / M[0, 0]
    >>> cr, cc
    (14.5, 14.5)
    """
    return moments_central(image, (0,) * image.ndim, order=order)


def moments_central(image, center=None, cc=None, order=3, **kwargs):
    """Calculate all central image moments up to a certain order.

    The center coordinates (cr, cc) can be calculated from the raw moments as:
    {``M[1, 0] / M[0, 0]``, ``M[0, 1] / M[0, 0]``}.

    Note that central moments are translation invariant but not scale and
    rotation invariant.

    Parameters
    ----------
    image : 2D double or uint8 array
        Rasterized shape as image.
    center : tuple of float, optional
        Coordinates of the image centroid. This will be computed if it
        is not provided.
    order : int, optional
        The maximum order of moments computed.

    Other Parameters
    ----------------
    cr : double
        DEPRECATED: Center row coordinate for 2D image.
    cc : double
        DEPRECATED: Center column coordinate for 2D image.

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

    Examples
    --------
    >>> image = np.zeros((20, 20), dtype=np.double)
    >>> image[13:17, 13:17] = 1
    >>> M = moments(image)
    >>> cr = M[1, 0] / M[0, 0]
    >>> cc = M[0, 1] / M[0, 0]
    >>> moments_central(image, (cr, cc))
    array([[ 16.,   0.,  20.,   0.],
           [  0.,   0.,   0.,   0.],
           [ 20.,   0.,  25.,   0.],
           [  0.,   0.,   0.,   0.]])
    """
    if cc is not None:  # using deprecated interface
        message = ('Using deprecated 2D-only, xy-coordinate interface to '
                   'moments_central. This interface will be removed in '
                   'scikit-image 0.16. Use '
                   'moments_central(image, center=(cr, cc), order=3).')
        warn(message)
        if 'cr' in kwargs and center is None:
            center = (kwargs['cr'], cc)
        else:
            center = (center, cc)
        return moments_central(image, center=center, order=order).T
    if center is None:
        M = moments_central(image, center=(0,) * image.ndim, order=order)
        center = M[tuple(np.eye(image.ndim, dtype=int))]
    calc = image.astype(float)
    for dim, dim_length in enumerate(image.shape):
        delta = np.arange(dim_length, dtype=float) - center[dim]
        powers_of_delta = delta[:, np.newaxis] ** np.arange(order + 1)
        calc = np.rollaxis(calc, dim, image.ndim)
        calc = np.dot(calc, powers_of_delta)
        calc = np.rollaxis(calc, -1, dim)
    return calc


def moments_normalized(mu, order=3):
    """Calculate all normalized central image moments up to a certain order.

    Note that normalized central moments are translation and scale invariant
    but not rotation invariant.

    Parameters
    ----------
    mu : (M,[ ...,] M) array
        Central image moments, where M must be greater than or equal
        to ``order``.
    order : int, optional
        Maximum order of moments. Default is 3.

    Returns
    -------
    nu : (``order + 1``,[ ...,] ``order + 1``) array
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

    Examples
    --------
    >>> image = np.zeros((20, 20), dtype=np.double)
    >>> image[13:17, 13:17] = 1
    >>> m = moments(image)
    >>> cr = m[0, 1] / m[0, 0]
    >>> cc = m[1, 0] / m[0, 0]
    >>> mu = moments_central(image, cr, cc)
    >>> moments_normalized(mu)
    array([[        nan,         nan,  0.078125  ,  0.        ],
           [        nan,  0.        ,  0.        ,  0.        ],
           [ 0.078125  ,  0.        ,  0.00610352,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ]])

    """
    if np.any(np.array(mu.shape) <= order):
        raise ValueError("Shape of image moments must be >= `order`")
    nu = np.zeros_like(mu)
    mu0 = mu.ravel()[0]
    for powers in itertools.product(range(order + 1), repeat=mu.ndim):
        if sum(powers) < mu.ndim:
            nu[powers] = np.nan
        else:
            nu[powers] = mu[powers] / (mu0 ** (sum(powers) / nu.ndim + 1))
    return nu


def moments_hu(nu):
    """Calculate Hu's set of image moments (2D-only).

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
    return _moments_cy.moments_hu(nu.astype(np.double))
