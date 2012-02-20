# Authors: Nicolas Poilvert <poilvert@rowland.harvard.edu>
#          Nicolas Pinto <pinto@alum.mit.edu>
# License: BSD 3-clause

import numpy as np
from scipy import ndimage as ndi


def resample(arr_in, out_shape, order=0, intp2d=True):
    """Resample (and possibly interpolate) a 3-dimensional array to the
    desired shape.

    Parameters
    ----------
    arr_in: array, shape = [height, width, depth]
        Input array

    out_shape: tuple of int
        Desired output shape after resampling.
        Format = [new_height, new_width, new_dept]

    order: int, optional
        Interpolation order. 0=nearest, 1=linear, 3=cubic, etc.
        Values between 0 and 5 are possible. Default is 0.

    intp2d: bool, optional
        If True, indicates that we only want to interpolate in 2d
        within every "slice" along the third dimension (depth).
        Default is True.

    Returns
    -------
    arr_out: array, shape = `out_shape`
        Resampled input array of shape `out_shape`.

    Example
    -------
    >>> import numpy as np
    >>> from pythor3.wildwest.operation import resample
    >>> X = np.random.randn(10, 10, 96)
    >>> X.shape
    >>> (10, 10, 96)
    >>> Y = resample(X, (5, 5, 96), intp2d=True)
    >>> Y.shape
    >>> (5, 5, 96)
    """

    # -- list of interpolation methods
    possible_orders = [0, 1, 2, 3, 4, 5]

    # -----------------
    # Checks on inputs
    # -----------------

    assert arr_in.ndim == 3
    assert order in possible_orders
    assert type(intp2d) == bool
    assert len(out_shape) == arr_in.ndim

    if intp2d:
        assert arr_in.shape[2] == out_shape[2]

    # -- parameters
    h_in, w_in, d_in = arr_in.shape
    h_out, w_out, d_out = out_shape

    # -----------------
    # 2D interpolation
    # -----------------
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
            slice2D = ndi.map_coordinates(arr_in[:, :, d],
                                          coordinates, order=order)
            arr_out[:, :, d] = slice2D

        # -- return output array
        return arr_out

    # -----------------
    # 3D interpolation
    # -----------------
    else:

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
