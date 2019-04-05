import numpy as np

from ._shared.utils import safe_as_int


def _stackcopy(a, b):
    """Copy b into each color layer of a, such that::

      a[:,:,0] = a[:,:,1] = ... = b

    Parameters
    ----------
    a : (M, N) or (M, N, P) ndarray
        Target array.
    b : (M, N)
        Source array.

    Notes
    -----
    Color images are stored as an ``(M, N, 3)`` or ``(M, N, 4)`` arrays.

    """
    if a.ndim == 3:
        a[:] = b[:, :, np.newaxis]
    else:
        a[:] = b


def warp_coords(coord_map, shape, dtype=np.float64):
    """Build the source coordinates for the output of a 2-D image warp.

    Parameters
    ----------
    coord_map : callable like GeometricTransform.inverse
        Return input coordinates for given output coordinates.
        Coordinates are in the shape (P, 2), where P is the number
        of coordinates and each element is a ``(row, col)`` pair.
    shape : tuple
        Shape of output image ``(rows, cols[, bands])``.
    dtype : np.dtype or string
        dtype for return value (sane choices: float32 or float64).

    Returns
    -------
    coords : (ndim, rows, cols[, bands]) array of dtype `dtype`
            Coordinates for `scipy.ndimage.map_coordinates`, that will yield
            an image of shape (orows, ocols, bands) by drawing from source
            points according to the `coord_transform_fn`.

    Notes
    -----

    This is a lower-level routine that produces the source coordinates for 2-D
    images used by `warp()`.

    It is provided separately from `warp` to give additional flexibility to
    users who would like, for example, to re-use a particular coordinate
    mapping, to use specific dtypes at various points along the the
    image-warping process, or to implement different post-processing logic
    than `warp` performs after the call to `ndi.map_coordinates`.


    Examples
    --------
    Produce a coordinate map that shifts an image up and to the right:

    >>> from skimage import data
    >>> from scipy.ndimage import map_coordinates
    >>>
    >>> def shift_up10_left20(xy):
    ...     return xy - np.array([-20, 10])[None, :]
    >>>
    >>> image = data.astronaut().astype(np.float32)
    >>> coords = warp_coords(shift_up10_left20, image.shape)
    >>> warped_image = map_coordinates(image, coords)

    """
    shape = safe_as_int(shape)
    rows, cols = shape[0], shape[1]
    coords_shape = [len(shape), rows, cols]
    if len(shape) == 3:
        coords_shape.append(shape[2])
    coords = np.empty(coords_shape, dtype=dtype)

    # Reshape grid coordinates into a (P, 2) array of (row, col) pairs
    tf_coords = np.indices((cols, rows), dtype=dtype).reshape(2, -1).T

    # Map each (row, col) pair to the source image according to
    # the user-provided mapping
    tf_coords = coord_map(tf_coords)

    # Reshape back to a (2, M, N) coordinate grid
    tf_coords = tf_coords.T.reshape((-1, cols, rows)).swapaxes(1, 2)

    # Place the y-coordinate mapping
    _stackcopy(coords[1, ...], tf_coords[0, ...])

    # Place the x-coordinate mapping
    _stackcopy(coords[0, ...], tf_coords[1, ...])

    if len(shape) == 3:
        coords[2, ...] = range(shape[2])

    return coords


def _clip_warp_output(input_image, output_image, order, mode, cval, clip):
    """Clip output image to range of values of input image.

    Note that this function modifies the values of `output_image` in-place
    and it is only modified if ``clip=True``.

    Parameters
    ----------
    input_image : ndarray
        Input image.
    output_image : ndarray
        Output image, which is modified in-place.

    Other parameters
    ----------------
    order : int, optional
        The order of the spline interpolation, default is 1. The order has to
        be in the range 0-5. See `skimage.transform.warp` for detail.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.

    """
    if clip and order != 0:
        min_val = input_image.min()
        max_val = input_image.max()

        preserve_cval = (mode == 'constant' and not
                         (min_val <= cval <= max_val))

        if preserve_cval:
            cval_mask = output_image == cval

        np.clip(output_image, min_val, max_val, out=output_image)

        if preserve_cval:
            output_image[cval_mask] = cval

