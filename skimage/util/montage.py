__all__ = ['montage2d']

import numpy as np
from .. import exposure

EPSILON = 1e-6


def montage2d(arr_in, fill='mean', rescale_intensity=False):
    """Create a 2-dimensional 'montage' from a 3-dimensional input array
    representing an ensemble of equally shaped 2-dimensional images.

    For example, montage2d(arr_in, fill) with the following `arr_in`

    +---+---+---+
    | 1 | 2 | 3 |
    +---+---+---+

    will return:

    +---+---+
    | 1 | 2 |
    +---+---+
    | 3 | * |
    +---+---+

    Where the '*' patch will be determined by the `fill` parameter.

    Parameters
    ----------
    arr_in: ndarray, shape=[n_images, height, width]
        3-dimensional input array representing an ensemble of n_images
        of equal shape (i.e. [height, width]).

    fill: float or 'mean', optional
        How to fill the 2-dimensional output array when sqrt(n_images)
        is not an integer. If 'mean' is chosen, then fill = arr_in.mean().

    rescale_intensity: bool, optional
        Whether to rescale the intensity of each image to [0, 1].

    Returns
    -------
    arr_out: ndarray, shape=[alpha * height, alpha * width]
        Output array where 'alpha' has been determined automatically to
        fit (at least) the `n_images` in `arr_in`.

    Example
    -------
    >>> import numpy as np
    >>> from skimage.util.montage import montage2d
    >>> arr_in = np.arange(3 * 2 * 2).reshape(3, 2, 2)
    >>> print arr_in  # doctest: +NORMALIZE_WHITESPACE
    [[[ 0  1]
      [ 2  3]]
     [[ 4  5]
      [ 6  7]]
     [[ 8  9]
      [10 11]]]
    >>> arr_out = montage2d(arr_in)
    >>> print arr_out.shape
    (4, 4)
    >>> print arr_out
    [[  0.    1.    4.    5. ]
     [  2.    3.    6.    7. ]
     [  8.    9.    5.5   5.5]
     [ 10.   11.    5.5   5.5]]
    >>> print arr_in.mean()
    5.5
    """
    assert arr_in.ndim == 3

    n_images, height, width = arr_in.shape

    # -- rescale intensity if necessary
    if rescale_intensity:
        for i in xrange(n_images):
            arr_in[i] = exposure.rescale_intensity(arr_in[i])

    # -- determine alpha
    alpha = int(np.ceil(np.sqrt(n_images)))

    # -- fill missing patches
    if fill == 'mean':
        fill = arr_in.mean()

    n_missing = int((alpha ** 2.) - n_images)
    missing = np.ones((n_missing, height, width), dtype=arr_in.dtype) * fill
    arr_out = np.vstack((arr_in, missing))

    # -- reshape to 2d montage, step by step
    arr_out = arr_out.reshape(alpha, alpha, height, width)
    arr_out = arr_out.swapaxes(1, 2)
    arr_out = arr_out.reshape(alpha * height, alpha * width)

    return arr_out
