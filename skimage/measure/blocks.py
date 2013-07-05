import numpy as np
from ..util.shape import view_as_blocks, _pad_asymmetric_zeros


def _block_func(image, factors, func):
    """Down-sample image by integer factors.

    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    factors : array_like
        Array containing down-sampling integer factor along each axis.
    func : object
        Function object which is used to calculate the return value for each
        local block, e.g. `numpy.sum`.

    Returns
    -------
    image : ndarray
        Down-sampled image with same number of dimensions as input image.

    """

    pad_size = []
    if len(factors) != image.ndim:
        raise ValueError("`factors` must have the same length "
                         "as `image.shape`.")

    for i in range(len(factors)):
        if image.shape[i] % factors[i] != 0:
            pad_size.append(factors[i] - (image.shape[i] % factors[i]))
        else:
            pad_size.append(0)

    for i in range(len(pad_size)):
        image = _pad_asymmetric_zeros(image, pad_size[i], i)

    out = view_as_blocks(image, factors)
    block_shape = out.shape

    for i in range(len(block_shape) // 2):
        out = func(out, axis=-1)

    return out


def block_sum(image, block_size):
    """Sum elements in local blocks.

    The image is padded with zeros if it is not perfectly divisible by integer
    factors.

    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    block_size : array_like
        Array containing down-sampling integer factor along each axis.

    Returns
    -------
    image : ndarray
        Down-sampled image with same number of dimensions as input image.

    Example
    -------
    >>> a = np.arange(15).reshape(3, 5)
    >>> a
    image([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])
    >>> block_sum(a, (2, 3))
    image([[21, 24],
           [33, 27]])

    """
    return _block_func(image, block_size, np.sum)


def block_mean(image, block_size):
    """Average elements in local blocks.

    The image is padded with zeros if it is not perfectly divisible by integer
    factors.

    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    block_size : array_like
        Array containing down-sampling integer factor along each axis.

    Returns
    -------
    image : ndarray
        Down-sampled image with same number of dimensions as input image.

    Example
    -------
    >>> a = np.arange(15).reshape(3, 5)
    >>> a
    image([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])
    >>> block_mean(a, (2, 3))
    array([[ 3.5,  4. ],
           [ 5.5,  4.5]])

    """
    return _block_func(image, block_size, np.mean)


def block_median(image, block_size):
    """Median element in local blocks.

    The image is padded with zeros if it is not perfectly divisible by integer
    factors.

    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    block_size : array_like
        Array containing down-sampling integer factor along each axis.

    Returns
    -------
    image : ndarray
        Down-sampled image with same number of dimensions as input image.

    Example
    -------
    >>> a = np.array([[1, 5, 100], [0, 5, 1000]])
    >>> a
    array([[   1,    5,  100],
           [   0,    5, 1000]])
    >>> block_median(a, (2, 3))
    array([[ 5.]])

    """
    return _block_func(image, block_size, np.median)


def block_min(image, block_size):
    """Minimum element in local blocks.

    The image is padded with zeros if it is not perfectly divisible by integer
    factors.

    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    block_size : array_like
        Array containing down-sampling integer factor along each axis.

    Returns
    -------
    image : ndarray
        Down-sampled image with same number of dimensions as input image.

    Example
    -------
    >>> a = np.arange(15).reshape(3, 5)
    >>> a
    image([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])
    >>> block_min(a, (2, 2))
    array([[0, 2, 0],
           [0, 0, 0]])

    """
    return _block_func(image, block_size, np.min)


def block_max(image, block_size):
    """Maximum element in local blocks.

    The image is padded with zeros if it is not perfectly divisible by integer
    factors.

    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    block_size : array_like
        Array containing down-sampling integer factor along each axis.

    Returns
    -------
    image : ndarray
        Down-sampled image with same number of dimensions as input image.

    Example
    -------
    >>> a = np.arange(15).reshape(3, 5)
    >>> a
    image([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])
    >>> block_max(a, (2, 3))
    array([[ 7,  9],
           [12, 14]])

    """
    return _block_func(image, block_size, np.max)
