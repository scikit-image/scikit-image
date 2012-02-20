import numpy as np
from scipy import ndimage as ndi

from _resample import _resample_float32


def resample(arr_in, out_shape, order=0, intp2d=False):
    """Resample a 3-dimensional array to the desired shape.


    Parameters
    ----------
    arr_in: ndarray, shape = [height, width, depth]
        Input array

    out_shape: tuple of int
        Desired output shape after resampling.
        Format = [new_height, new_width, new_depth]

    order: int, optional
        Interpolation order. 0=nearest, 1=linear, 3=cubic, etc.
        Values between 0 and 5 are possible. Default is 0.

    intp2d: bool, optional
        If True, indicates that we only want to interpolate in 2d
        within every "slice" along the third dimension (depth).
        Default is False.

    Returns
    -------
    arr_out: ndarray, shape = `out_shape`
        Resampled input array of shape `out_shape`.

    Example
    -------
    >>> import numpy as np
    >>> from skimage.shape import resample
    >>> X = np.random.randn(10, 10, 4)
    >>> X.shape
    (10, 10, 4)
    >>> Y = resample(X, (5, 5, 4), intp2d=True)
    >>> Y.shape
    (5, 5, 4)
    >>> X = np.random.randn(256, 256, 96).astype('f')
    >>> X.shape
    (256, 256, 96)
    >>> Y = resample(X, (512, 512, 128))
    >>> Y.shape
    (512, 512, 128)
    """

    # -- list of interpolation methods
    possible_orders = range(5 + 1)

    # -------------------------------------------------------------------------
    # -- Check arguments
    # -------------------------------------------------------------------------
    if arr_in.ndim != 3:
        raise ValueError('input array should be 3 dimensional')
    if order not in possible_orders:
        raise ValueError('interpolation order unsupported')
    if type(intp2d) != bool:
        raise ValueError("kwarg 'intp2d' is expected to be of bool type")
    if len(out_shape) != arr_in.ndim:
        raise ValueError('cannot resample array to higher or lower '
                         'dimensionality'
                        )

    if intp2d and arr_in.shape[2] != out_shape[2]:
        raise ValueError(
            "intp2d cannot be True if the arguments don't have the same depth "
            "(i.e. arr_in.shape[2] must be equal to out_shape[2])"
            )

    h_in, w_in, d_in = arr_in.shape
    h_out, w_out, d_out = out_shape

    # -------------------------------------------------------------------------
    # -- Special case where order=0 (i.e. interpolation 'nearest')
    # -------------------------------------------------------------------------
    # We use Cython for faster processing. Only float32 is supported for now.
    # XXX: (WIP) template cython code to be compatible with all dtypes
    if arr_in.dtype == np.float32 and order == 0:
        # prepare output array
        arr_out = np.empty(out_shape, dtype=arr_in.dtype)
        _resample_float32(arr_in, arr_out)
        return arr_out

    # -------------------------------------------------------------------------
    # -- 2D interpolation(s)
    # -------------------------------------------------------------------------
    if intp2d:

        # -- initialize output array
        arr_out = np.empty(out_shape, dtype=arr_in.dtype)

        # -- output grid in the first two dimensions
        h_out_grid, w_out_grid = np.mgrid[:h_out, :w_out]

        # -- rescaling of the grids to input array range
        h_out_grid = (1. * h_out_grid / h_out_grid.max()) * (h_in - 1.)
        w_out_grid = (1. * w_out_grid / w_out_grid.max()) * (w_in - 1.)

        # -- coordinates of the output array pixels
        coordinates = np.array([h_out_grid, w_out_grid])

        # -- loop over the third dimension (2D interpolation)
        for d in xrange(int(d_out)):
            slice2D = ndi.map_coordinates(
                arr_in[:, :, d], coordinates, order=order)
            arr_out[:, :, d] = slice2D

        # -- return output array
        return arr_out

    # -------------------------------------------------------------------------
    # -- 3D interpolation
    # -------------------------------------------------------------------------
    # -- output grid
    h_out_grid, w_out_grid, d_out_grid = np.mgrid[:h_out, :w_out, :d_out]

    # -- rescaling of the grids to input array range
    h_out_grid = (1. * h_out_grid / h_out_grid.max()) * (h_in - 1.)
    w_out_grid = (1. * w_out_grid / w_out_grid.max()) * (w_in - 1.)
    d_out_grid = (1. * d_out_grid / d_out_grid.max()) * (d_in - 1.)

    # -- interpolation
    coordinates = np.array([h_out_grid, w_out_grid, d_out_grid])
    arr_out = ndi.map_coordinates(arr_in, coordinates, order=order)

    # -- return output array
    return arr_out
