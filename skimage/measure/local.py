import numpy as np
from skimage.util import view_as_blocks, pad


def _local_func(image, block_size, func, cval):
    """Down-sample image by applying function to local blocks.

    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    block_size : array_like
        Array containing down-sampling integer factor along each axis.
    func : object
        Function object which is used to calculate the return value for each
        local block, e.g. `numpy.sum`.
    cval : float, optional
        Constant padding value if image is not perfectly divisible by the
        block size.

    Returns
    -------
    image : ndarray
        Down-sampled image with same number of dimensions as input image.

    """

    if len(block_size) != image.ndim:
        raise ValueError("`block_size` must have the same length "
                         "as `image.shape`.")

    pad_width = []
    for i in range(len(block_size)):
        if image.shape[i] % block_size[i] != 0:
            after_width = block_size[i] - (image.shape[i] % block_size[i])
        else:
            after_width = 0
        pad_width.append((0, after_width))

    image = pad(image, pad_width=pad_width, mode='constant',
                constant_values=cval)

    out = view_as_blocks(image, block_size)

    for i in range(len(out.shape) // 2):
        out = func(out, axis=-1)

    return out


def local_sum(image, block_size, cval=0):
    """Sum elements in local blocks.

    The image is padded with zeros if it is not perfectly divisible by the
    block size.

    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    block_size : array_like
        Array containing down-sampling integer factor along each axis.
    cval : float, optional
        Constant padding value if image is not perfectly divisible by the
        block size.

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
    return _local_func(image, block_size, np.sum, cval)


def local_mean(image, block_size, cval=0):
    """Average elements in local blocks.

    The image is padded with zeros if it is not perfectly divisible by the
    block size.

    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    block_size : array_like
        Array containing down-sampling integer factor along each axis.
    cval : float, optional
        Constant padding value if image is not perfectly divisible by the
        block size.

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
    return _local_func(image, block_size, np.mean, cval)


def local_median(image, block_size, cval=0):
    """Median element in local blocks.

    The image is padded with zeros if it is not perfectly divisible by the
    block size.

    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    block_size : array_like
        Array containing down-sampling integer factor along each axis.
    cval : float, optional
        Constant padding value if image is not perfectly divisible by the
        block size.

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
    return _local_func(image, block_size, np.median, cval)


def local_min(image, block_size, cval=0):
    """Minimum element in local blocks.

    The image is padded with zeros if it is not perfectly divisible by the
    block size.

    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    block_size : array_like
        Array containing down-sampling integer factor along each axis.
    cval : float, optional
        Constant padding value if image is not perfectly divisible by the
        block size.

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
    return _local_func(image, block_size, np.min, cval)


def local_max(image, block_size, cval=0):
    """Maximum element in local blocks.

    The image is padded with zeros if it is not perfectly divisible by the
    block size.

    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    block_size : array_like
        Array containing down-sampling integer factor along each axis.
    cval : float, optional
        Constant padding value if image is not perfectly divisible by the
        block size.

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
    return _local_func(image, block_size, np.max, cval)
