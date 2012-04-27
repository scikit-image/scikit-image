__all__ = ['warp']

import numpy as np
from scipy import ndimage
from skimage.util import img_as_float

eps = np.finfo(float).eps

def _stackcopy(a, b):
    """a[:,:,0] = a[:,:,1] = ... = b"""
    if a.ndim == 3:
        a.transpose().swapaxes(1, 2)[:] = b
    else:
        a[:] = b

def warp(image, coord_tf, tf_args={},
         output_shape=None, order=1, mode='constant', cval=0.):
    """Warp an image according to a given coordinate transformation.

    Parameters
    ----------
    image : 2-D array
        Input image.
    coord_tf : callable xy = f(xy, **kwargs)
        Function that transforms an Nx2 array of ``(x, y)`` coordinates
        in the *output image* into their corresponding coordinates in the
        *source image*.  Note that this is a reverse mapping (also
        see examples below).
    tf_args : dict, optional
        Keyword arguments passed to `coord_tf`.
    output_shape : tuple (rows, cols)
        Shape of the output image generated.
    order : int
        Order of splines used in interpolation.
    mode : string
        How to handle values outside the image borders.  Passed as-is
        to ndimage.
    cval : string
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Examples
    --------
    Shift an image to the right:

    >>> from skimage import data
    >>> image = data.camera()
    >>>
    >>> def shift_right(xy):
    ...     xy[:, 0] -= 10
    ...     return xy
    >>>
    >>> warp(image, shift_right)

    """
    if image.ndim < 2:
        raise ValueError("Input must have more than 1 dimension.")

    image = np.atleast_3d(img_as_float(image))
    ishape = np.array(image.shape)
    bands = ishape[2]

    if output_shape is None:
        output_shape = ishape

    coords = np.empty(np.r_[3, output_shape], dtype=float)

    # Construct transformed coordinates
    rows, cols = output_shape[:2]
    tf_coords = np.indices((cols, rows), dtype=float).reshape(2, -1).T

    tf_coords = coord_tf(tf_coords, **tf_args)
    tf_coords = tf_coords.T.reshape((-1, cols, rows)).swapaxes(1, 2)

    # y-coordinate mapping
    _stackcopy(coords[1, ...], tf_coords[0, ...])

    # x-coordinate mapping
    _stackcopy(coords[0, ...], tf_coords[1, ...])

    # colour-coordinate mapping
    coords[2, ...] = range(bands)

    # Prefilter not necessary for order 1 interpolation
    prefilter = order > 1
    mapped = ndimage.map_coordinates(image, coords, prefilter=prefilter,
                                     mode=mode, order=order, cval=cval)

    # The spline filters sometimes return results outside [0, 1],
    # so clip to ensure valid data
    return np.clip(mapped.squeeze(), 0, 1)
