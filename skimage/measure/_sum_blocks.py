from ..transform._warps import _downsample


def sum_blocks(array, factors):
	"""Sums the elements in blocks of integer factors and pads the original
	array with zeroes if the dimensions are not perfectly divisible by factors.

    Parameters
    ----------
    array : ndarray
        Input n-dimensional array.
    factors: tuple
        Tuple containing integer values representing block length along each
        axis.

    Returns
    -------
    array : ndarray
        Downsampled array with same number of dimensions as that of input
        array.

    Example
    -------
    >>> a = np.arange(15).reshape(3, 5)
    >>> a
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])
    >>> sum_blocks(a, (2,3))
    array([[21, 24],
           [33, 27]])

    """
	return _downsample(array, factors)
