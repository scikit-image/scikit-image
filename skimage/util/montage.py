__all__ = ['montage2d']

import numpy as np
from .. import exposure

EPSILON = 1e-6


def montage2d(arr_in, fill='mean', rescale_intensity=False, grid_shape=None, padding_width=0):
    """Create a 2-dimensional 'montage' from a 3-dimensional input array
    representing an ensemble of equally shaped 2-dimensional images.

    For example, ``montage2d(arr_in, fill)`` with the following `arr_in`

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
    arr_in : ndarray, shape=[n_images, height, width]
        3-dimensional input array representing an ensemble of n_images
        of equal shape (i.e. [height, width]).
    fill : float or 'mean', optional
        How to fill the 2-dimensional output array when sqrt(n_images)
        is not an integer. If 'mean' is chosen, then fill = arr_in.mean().
    rescale_intensity : bool, optional
        Whether to rescale the intensity of each image to [0, 1].
    grid_shape : tuple, optional
        The desired grid shape for the montage (tiles_y, tiles_x).
        The default aspect ratio is square.
    padding_width : int, optional
        The size of the spacing between the tiles to make the
        boundaries of individual frames easier to see.

    Returns
    -------
    arr_out : ndarray, shape=[alpha * height, alpha * width]
        Output array where 'alpha' has been determined automatically to
        fit (at least) the `n_images` in `arr_in`.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.util.montage import montage2d
    >>> arr_in = np.arange(3 * 2 * 2).reshape(3, 2, 2)
    >>> arr_in.astype(int)  # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0,  1],
            [ 2,  3]],
           [[ 4,  5],
            [ 6,  7]],
           [[ 8,  9],
            [10, 11]]])
    >>> arr_out = montage2d(arr_in)
    >>> arr_out.shape
    (4, 4)
    >>> arr_out
    array([[  0. ,   1. ,   4. ,   5. ],
           [  2. ,   3. ,   6. ,   7. ],
           [  8. ,   9. ,   5.5,   5.5],
           [ 10. ,  11. ,   5.5,   5.5]])
    >>> arr_in.mean()
    5.5
    >>> arr_out_nonsquare = montage2d(arr_in, grid_shape=(1, 3))
    >>> arr_out_nonsquare.astype(int)
    array([[  0,   1,   4,   5,   8,   9],
           [  2,   3,   6,   7,  10,  11]])
    >>> arr_out_nonsquare.shape
    (2, 6)

    """

    assert arr_in.ndim == 3

    # -- fill missing patches (needs to be calculated before border padding)
    if fill == 'mean':
        fill = arr_in.mean()

    # -- add border padding, np.pad does all dimensions
    # so we remove the padding from the first
    if padding_width > 0:
        # only pad after to make the width correct
        bef_aft = (0, padding_width)
        arr_in = np.pad(arr_in, ((0,0), bef_aft, bef_aft), mode='constant')
    else:
        arr_in = arr_in.copy()

    n_images, height, width = arr_in.shape

    # -- rescale intensity if necessary
    if rescale_intensity:
        for i in range(n_images):
            arr_in[i] = exposure.rescale_intensity(arr_in[i])

    # -- determine alpha
    if grid_shape:
        alpha_y, alpha_x = grid_shape
    else:
        alpha_y = alpha_x = int(np.ceil(np.sqrt(n_images)))



    n_missing = int((alpha_y * alpha_x) - n_images)
    # sometimes the mean returns a float, this ensures the missing
    # has the same type for non-float images
    missing = (np.ones((n_missing, height, width), dtype=arr_in.dtype) * fill).astype(arr_in.dtype)
    arr_out = np.vstack((arr_in, missing))

    # -- reshape to 2d montage, step by step
    arr_out = arr_out.reshape(alpha_y, alpha_x, height, width)
    arr_out = arr_out.swapaxes(1, 2)
    arr_out = arr_out.reshape(alpha_y * height, alpha_x * width)

    return arr_out

def montage_rgb(arr_in, fill='mean', grid_shape=None, padding_width=0):
    """Create a 3-dimensional 'montage' from a 4-dimensional input array
    representing an ensemble of equally shaped 3-dimensional images.


    Parameters
    ----------
    arr_in : ndarray, shape=[n_images, height, width, n_channels]
        3-dimensional input array representing an ensemble of n_images
        of equal shape (i.e. [height, width, n_channels]).
    fill : float or 'mean', optional
        How to fill the 2-dimensional output array when sqrt(n_images)
        is not an integer. If 'mean' is chosen, then fill = arr_in.mean().
    grid_shape : tuple, optional
        The desired grid shape for the montage (tiles_y, tiles_x).
        The default aspect ratio is square.
    border_padding : int, optional
        The size of the spacing between the tiles to make the
        boundaries of individual frames easier to see.

    Returns
    -------
    arr_out : ndarray, shape=[alpha * height, alpha * width, n_channels]
        Output array where 'alpha' has been determined automatically to
        fit (at least) the `n_images` in `arr_in`.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.util.montage import montage_rgb
    >>> arr_in = np.arange(3 * 2 * 2 * 3).reshape(3, 2, 2, 3)
    >>> arr_out = montage_rgb(arr_in)
    >>> arr_out.shape
    (4, 4, 3)
    >>> arr_out.astype(int) # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [12, 13, 14],
        [15, 16, 17]],
       [[ 6,  7,  8],
        [ 9, 10, 11],
        [18, 19, 20],
        [21, 22, 23]],
       [[24, 25, 26],
        [27, 28, 29],
        [16, 17, 18],
        [16, 17, 18]],
       [[30, 31, 32],
        [33, 34, 35],
        [16, 17, 18],
        [16, 17, 18]]])
    """
    assert arr_in.ndim == 4

    n_images, height, width, n_channels = arr_in.shape

    out_slices = []
    for i_chan in range(n_channels):
        out_slices += [montage2d(arr_in[:,:,:,i_chan], fill=fill,
                                 grid_shape=grid_shape, padding_width=padding_width)]


    return np.stack(out_slices, 2)
