# Authors: Nicolas Poilvert <nicolas.poilvert@gmail.com>
#          Nicolas Pinto <nicolas.pinto@gmail.com>
# License: BSD 3-clause

__all__ = ['block_view', 'rolling_view']

import numpy as np
from numpy.lib.stride_tricks import as_strided as ast


def block_view(arr, block):
    """
    Offers a view on array 'arr' which allows one to easily
    pick a 'block' and reason within that block when
    manipulating the array indices.

    Parameters
    ----------
    arr: ndarray
        input array from which we want to obtain a
        block view

    block: tuple
        each element in the tuple represents the number of
        input array elements to include in a block along
        the corresponding direction

    Returns
    -------
    block view on input array.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.arange(4*4).reshape(4,4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> B = block_view(A, block=(2,2))
    >>> B[0, 1]
    array([[2, 3],
           [6, 7]])
    >>> B[1, 0, 1, 1]
    13
    >>> A = np.arange(4*4*6).reshape(4,4,6)
    >>> A
    array([[[ 0,  1,  2,  3,  4,  5],
            [ 6,  7,  8,  9, 10, 11],
            [12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23]],

           [[24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35],
            [36, 37, 38, 39, 40, 41],
            [42, 43, 44, 45, 46, 47]],

           [[48, 49, 50, 51, 52, 53],
            [54, 55, 56, 57, 58, 59],
            [60, 61, 62, 63, 64, 65],
            [66, 67, 68, 69, 70, 71]],

           [[72, 73, 74, 75, 76, 77],
            [78, 79, 80, 81, 82, 83],
            [84, 85, 86, 87, 88, 89],
            [90, 91, 92, 93, 94, 95]]])
    >>> B = block_view(A, block=(1,2,2))
    >>> B.shape
    >>> (4, 2, 3, 1, 2, 2)
    >>> B[2:, 0, 2]
    array([[[[52, 53],
             [58, 59]]],

           [[[76, 77],
             [82, 83]]]])
    """

    # -- if 'block' is None, we simply return the
    #    original array.
    if block == None:
        return arr
    # -- otherwise we make sure the user gave a
    #    tuple
    if not isinstance(block, tuple):
        raise ValueError('block needs to be a tuple')

    # -- basic invalid values for 'block'
    block_shape = np.array(block).astype(np.int)
    if (block_shape <= 0).any():
        raise ValueError('non strictly positive block shape given')
    if block_shape.size > arr.ndim:
        raise ValueError('block ndim larger than input array ndim')
    if block_shape.size < arr.ndim:
        raise ValueError('block ndim smaller than input array ndim')

    # -- checking that the block view is compatible
    #    with the shape of the input array
    A_shape = np.array(arr.shape).astype(np.int)
    if (A_shape % block_shape).sum() != 0:
        raise ValueError('block shape not compatible with input array')

    # -- actually building the block view
    rng = range(len(block))
    shape = (
        tuple([arr.shape[i] / block[i] for i in rng])
        + block
        )
    strides = (
        tuple([arr.strides[i] * block[i] for i in rng])
        + arr.strides
        )

    return ast(arr, shape=shape, strides=strides)


def rolling_view(arr, window_shape):
    """
    This function offers a 'rolling view' for any N-dimensional
    array. The 'window' defines the shape of the elementary
    N-dimensional orthotope (better know as hyperrectangle [1])
    of the view.

    Parameters
    ----------
    arr: ndarray object
        N-dimensional input array

    window_shape: N-tuple
        tuple of size N that gives the shape of the elementary
        window

    Returns
    -------
    a rolling view on the input array

    Notes
    -----
    One should be very careful with rolling views when it comes to
    memory usage. Indeed, although a 'view' has the same memory
    footprint as its base array, the actual array that emerges when
    this 'view' is used in a computation is generally a (much)
    larger array than the original, especially for 2-dimensional
    arrays and above.

    For example, let us consider a 3 dimensional array of size
    (100, 100, 100) of ``float64``. This array takes about 8*100**3
    Bytes for storage which is just 8 MB. If one decides to build
    a rolling view on this array with a window of (3, 3, 3) the
    hypothetical size of the rolling view (if one was to reshape
    the view for example) would be 8*(100-3+1)**3*3**3 which is
    about 203 MB! The scaling becomes even worse as the dimension
    of the input array becomes larger.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Hyperrectangle

    Examples
    --------
    >>> A = np.arange(10)
    >>> A
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> window_shape = (3,)
    >>> B = rolling_view(A, window_shape)
    >>> B.shape
    (8, 3)
    >>> B
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6],
           [5, 6, 7],
           [6, 7, 8],
           [7, 8, 9]])
    >>> A = np.arange(5*4).reshape(5, 4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19]])
    >>> window_shape = (4, 3)
    >>> B = rolling_view(A, window_shape)
    >>> B.shape
    (2, 2, 4, 3)
    >>> B
    array([[[[ 0,  1,  2],
             [ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14]],

            [[ 1,  2,  3],
             [ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15]]],


           [[[ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14],
             [16, 17, 18]],

            [[ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15],
             [17, 18, 19]]]])
    """

    # -- basic requirements on inputs
    if not isinstance(arr, np.ndarray):
        raise ValueError('the input should be an ndarray object')
    if not isinstance(window_shape, tuple):
        raise ValueError('the window shape should be a tuple')
    if not (len(window_shape) == arr.ndim):
        raise ValueError('array dimension and window length dont match')

    # -- input array dimension
    N = arr.ndim

    # -- compatibility checks
    if ((np.array(arr.shape).astype(int) - \
         np.array(window_shape).astype(int)) < 0).any():
        raise ValueError('window shape is too large')

    if ((np.array(window_shape).astype(int) - \
         np.ones(N).astype(int)) < 0).any():
        raise ValueError('window shape is too small')

    # -- shape of output 'rolling view' array
    out_shape = tuple([arr.shape[i] - window_shape[i] + 1
                       for i in range(N)]) + \
                window_shape

    # -- strides of output 'rolling view' array
    out_strides = arr.strides + arr.strides

    return ast(arr, shape=out_shape, strides=out_strides)
