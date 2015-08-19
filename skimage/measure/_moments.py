# coding: utf-8
import numpy as np
from . import _moments_cy


def moments(image, order=3):
    """Calculate all raw image moments up to a certain order.

    The following properties can be calculated from raw image moments:
     * Area as: ``m[0, 0]``.
     * Centroid as: {``m[0, 1] / m[0, 0]``, ``m[1, 0] / m[0, 0]``}.

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
    >>> m = moments(image)
    >>> cr = m[0, 1] / m[0, 0]
    >>> cc = m[1, 0] / m[0, 0]
    >>> cr, cc
    (14.5, 14.5)

    """
    return _moments_cy.moments_central(image, 0, 0, order)


def moments_central(image, cr, cc, order=3):
    """Calculate all central image moments up to a certain order.

    The center coordinates (cr, cc) can be calculated from the raw moments as:
    {``m[0, 1] / m[0, 0]``, ``m[1, 0] / m[0, 0]``}.

    Note that central moments are translation invariant but not scale and
    rotation invariant.

    Parameters
    ----------
    image : 2D double or uint8 array
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

    Examples
    --------
    >>> image = np.zeros((20, 20), dtype=np.double)
    >>> image[13:17, 13:17] = 1
    >>> m = moments(image)
    >>> cr = m[0, 1] / m[0, 0]
    >>> cc = m[1, 0] / m[0, 0]
    >>> moments_central(image, cr, cc)
    array([[ 16.,   0.,  20.,   0.],
           [  0.,   0.,   0.,   0.],
           [ 20.,   0.,  25.,   0.],
           [  0.,   0.,   0.,   0.]])
    """

    return _moments_cy.moments_central(image, cr, cc, order)


def moments_normalized(mu, order=3):
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
    if mu.ndim != 2:
        raise TypeError("Image moments must be 2-dimension")
    if mu.shape[0] <= order or mu.shape[1] <= order:
        raise TypeError("Shape of image moments must be >= `order`")
    return _moments_cy.moments_normalized(mu.astype(np.double), order)


def moments_hu(nu):
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
    return _moments_cy.moments_hu(nu.astype(np.double))
