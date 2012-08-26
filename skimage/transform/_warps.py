import numpy as np
from scipy import ndimage
from skimage.util import img_as_float
from ._geometric import (SimilarityTransform, AffineTransform,
                         ProjectiveTransform)
from ._warps_cy import _warp_fast


HOMOGRAPHY_TRANSFORMS = (
    SimilarityTransform,
    AffineTransform,
    ProjectiveTransform
)


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
    """Build the source coordinates for the output pixels of an image warp.

    Parameters
    ----------
    coord_map : callable like GeometricTransform.inverse
        Return input coordinates for given output coordinates.
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
    This is a lower-level routine that produces the source coordinates used by
    `warp()`.

    It is provided separately from `warp` to give additional flexibility to
    users who would like, for example, to re-use a particular coordinate
    mapping, to use specific dtypes at various points along the the
    image-warping process, or to implement different post-processing logic
    than `warp` performs after the call to `ndimage.map_coordinates`.


    Examples
    --------
    Produce a coordinate map that Shifts an image to the right:

    >>> from skimage import data
    >>> from scipy.ndimage import map_coordinates
    >>>
    >>> def shift_right(xy):
    ...     xy[:, 0] -= 10
    ...     return xy
    >>>
    >>> image = data.lena().astype(np.float32)
    >>> coords = warp_coords(shift_right, image.shape)
    >>> warped_image = map_coordinates(image, coords)

    """
    rows, cols = shape[0], shape[1]
    coords_shape = [len(shape), rows, cols]
    if len(shape) == 3:
        coords_shape.append(shape[2])
    coords = np.empty(coords_shape, dtype=dtype)

    # Reshape grid coordinates into a (P, 2) array of (x, y) pairs
    tf_coords = np.indices((cols, rows), dtype=dtype).reshape(2, -1).T

    # Map each (x, y) pair to the source image according to
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


def warp(image, inverse_map=None, map_args={}, output_shape=None, order=1,
         mode='constant', cval=0., reverse_map=None):
    """Warp an image according to a given coordinate transformation.

    Parameters
    ----------
    image : 2-D array
        Input image.
    inverse_map : transformation object, callable xy = f(xy, **kwargs)
        Inverse coordinate map. A function that transforms a (N, 2) array of
        ``(x, y)`` coordinates in the *output image* into their corresponding
        coordinates in the *source image* (e.g. a transformation object or its
        inverse).
    map_args : dict, optional
        Keyword arguments passed to `inverse_map`.
    output_shape : tuple (rows, cols)
        Shape of the output image generated.
    order : int
        Order of splines used in interpolation. See
        `scipy.ndimage.map_coordinates` for detail.
    mode : string
        How to handle values outside the image borders.  See
        `scipy.ndimage.map_coordinates` for detail.
    cval : float
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
    # Backward API compatibility
    if reverse_map is not None:
        inverse_map = reverse_map

    if image.ndim < 2:
        raise ValueError("Input must have more than 1 dimension.")

    orig_ndim = image.ndim
    image = np.atleast_3d(img_as_float(image))
    ishape = np.array(image.shape)
    bands = ishape[2]

    # use fast Cython version for specific parameters
    fast_modes = ('constant', 'reflect', 'wrap')
    if order in (0, 1) and mode in fast_modes and not map_args:
        matrix = None
        if isinstance(inverse_map, HOMOGRAPHY_TRANSFORMS):
            matrix = inverse_map._matrix
        elif inverse_map.__name__ == 'inverse' \
                and inverse_map.im_class in HOMOGRAPHY_TRANSFORMS:
            matrix = np.linalg.inv(inverse_map.im_self._matrix)
        if matrix is not None:
            # transform all bands
            dims = []
            for dim in range(image.shape[2]):
                dims.append(_warp_fast(image[..., dim], matrix,
                            output_shape=output_shape,
                            order=order, mode=mode, cval=cval))
            out = np.dstack(dims)
            if orig_ndim == 2:
                out = out[..., 0]
            return out

    if output_shape is None:
        output_shape = ishape

    rows, cols = output_shape[:2]

    def coord_map(*args):
        return inverse_map(*args, **map_args)

    coords = warp_coords(coord_map, (rows, cols, bands))

    # Prefilter not necessary for order 1 interpolation
    prefilter = order > 1
    mapped = ndimage.map_coordinates(image, coords, prefilter=prefilter,
                                     mode=mode, order=order, cval=cval)

    # The spline filters sometimes return results outside [0, 1],
    # so clip to ensure valid data
    clipped = np.clip(mapped, 0, 1)

    if mode == 'constant' and not (0 <= cval <= 1):
        clipped[mapped == cval] = cval

    # Remove singleton dim introduced by atleast_3d
    return clipped.squeeze()


def _swirl_mapping(xy, center, rotation, strength, radius):
    x, y = xy.T
    x0, y0 = center
    rho = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    # Ensure that the transformation decays to approximately 1/1000-th
    # within the specified radius.
    radius = radius / 5 * np.log(2)

    theta = rotation + strength * \
            np.exp(-rho / radius) + \
            np.arctan2(y - y0, x - x0)

    xy[..., 0] = x0 + rho * np.cos(theta)
    xy[..., 1] = y0 + rho * np.sin(theta)

    return xy


def swirl(image, center=None, strength=1, radius=100, rotation=0,
          output_shape=None, order=1, mode='constant', cval=0):
    """Perform a swirl transformation.

    Parameters
    ----------
    image : ndarray
        Input image.
    center : (x,y) tuple or (2,) ndarray
        Center coordinate of transformation.
    strength : float
        The amount of swirling applied.
    radius : float
        The extent of the swirl in pixels.  The effect dies out
        rapidly beyond `radius`.
    rotation : float
        Additional rotation applied to the image.

    Returns
    -------
    swirled : ndarray
        Swirled version of the input.

    Other parameters
    ----------------
    output_shape : tuple or ndarray
        Size of the generated output image.
    order : int
        Order of splines used in interpolation.  See
        `scipy.ndimage.map_coordinates` for detail.
    mode : string
        How to handle values outside the image borders.  See
        `scipy.ndimage.map_coordinates` for detail.
    cval : string
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    """

    if center is None:
        center = np.array(image.shape)[:2] / 2

    warp_args = {'center': center,
                 'rotation': rotation,
                 'strength': strength,
                 'radius': radius}

    return warp(image, _swirl_mapping, map_args=warp_args,
                output_shape=output_shape,
                order=order, mode=mode, cval=cval)


def homography(image, H, output_shape=None, order=1,
               mode='constant', cval=0.):
    """
    .. deprecated::
        0.7

    Perform a projective transformation (homography) on an image.

    For each pixel, given its homogeneous coordinate :math:`\mathbf{x}
    = [x, y, 1]^T`, its target position is calculated by multiplying
    with the given matrix, :math:`H`, to give :math:`H \mathbf{x}`.
    E.g., to rotate by theta degrees clockwise, the matrix should be

    ::

      [[cos(theta) -sin(theta) 0]
       [sin(theta)  cos(theta) 0]
       [0            0         1]]

    or, to translate x by 10 and y by 20,

    ::

      [[1 0 10]
       [0 1 20]
       [0 0 1 ]].

    Parameters
    ----------
    image : 2-D array
        Input image.
    H : array of shape ``(3, 3)``
        Transformation matrix H that defines the homography.
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
    >>> # rotate by 90 degrees around origin and shift down by 2
    >>> x = np.arange(9, dtype=np.uint8).reshape((3, 3)) + 1
    >>> x
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]], dtype=uint8)
    >>> theta = -np.pi/2
    >>> M = np.array([[np.cos(theta),-np.sin(theta),0],
    ...               [np.sin(theta), np.cos(theta),2],
    ...               [0,             0,            1]])
    >>> x90 = homography(x, M, order=1)
    >>> x90
    array([[3, 6, 9],
           [2, 5, 8],
           [1, 4, 7]], dtype=uint8)
    >>> # translate right by 2 and down by 1
    >>> y = np.zeros((5,5), dtype=np.uint8)
    >>> y[1, 1] = 255
    >>> y
    array([[  0,   0,   0,   0,   0],
           [  0, 255,   0,   0,   0],
           [  0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0]], dtype=uint8)
    >>> M = np.array([[ 1.,  0.,  2.],
    ...               [ 0.,  1.,  1.],
    ...               [ 0.,  0.,  1.]])
    >>> y21 = homography(y, M, order=1)
    >>> y21
    array([[  0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0],
           [  0,   0,   0, 255,   0],
           [  0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0]], dtype=uint8)

    """
    import warnings
    warnings.warn('the homography function is deprecated; '
                  'use the `warp` and `ProjectiveTransform` class instead',
                  category=DeprecationWarning)

    tform = ProjectiveTransform(H)
    return warp(image, inverse_map=tform.inverse, output_shape=output_shape,
                order=order, mode=mode, cval=cval)
