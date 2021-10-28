#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from scipy import ndimage as ndi
import numpy as np

from ._geometric import (SimilarityTransform, AffineTransform,
                         ProjectiveTransform)
from .utils import (get_bound_method_class, _to_ndimage_mode,
                    _validate_interpolation_order, convert_to_float,
                    safe_as_int, warn)

cimport numpy as cnp

from .interpolation cimport (nearest_neighbour_interpolation,
                             bilinear_interpolation,
                             biquadratic_interpolation,
                             bicubic_interpolation)
from .fused_numerics cimport np_floats

cnp.import_array()


HOMOGRAPHY_TRANSFORMS = (
    SimilarityTransform,
    AffineTransform,
    ProjectiveTransform
)


cdef inline void _transform_metric(np_floats x, np_floats y, np_floats* H,
                                   np_floats *x_, np_floats *y_) nogil:
    """Apply a metric transformation to a coordinate.

    Parameters
    ----------
    x, y : np_floats
        Input coordinate.
    H : (3,3) *np_floats
        Transformation matrix.
    x_, y_ : *np_floats
        Output coordinate.

    """
    x_[0] = H[0] * x + H[2]
    y_[0] = H[4] * y + H[5]


cdef inline void _transform_affine(np_floats x, np_floats y, np_floats* H,
                                   np_floats *x_, np_floats *y_) nogil:
    """Apply an affine transformation to a coordinate.

    Parameters
    ----------
    x, y : np_floats
        Input coordinate.
    H : (3,3) *np_floats
        Transformation matrix.
    x_, y_ : *np_floats
        Output coordinate.

    """
    x_[0] = H[0] * x + H[1] * y + H[2]
    y_[0] = H[3] * x + H[4] * y + H[5]


cdef inline void _transform_projective(np_floats x, np_floats y, np_floats* H,
                                       np_floats *x_, np_floats *y_) nogil:
    """Apply a homography to a coordinate.

    Parameters
    ----------
    x, y : np_floats
        Input coordinate.
    H : (3,3) *np_floats
        Transformation matrix.
    x_, y_ : *np_floats
        Output coordinate.

    """
    cdef np_floats z_
    z_ = H[6] * x + H[7] * y + H[8]
    x_[0] = (H[0] * x + H[1] * y + H[2]) / z_
    y_[0] = (H[3] * x + H[4] * y + H[5]) / z_


def _warp_fast(np_floats[:, :] image, np_floats[:, :] H, output_shape=None,
               int order=1, mode='constant', np_floats cval=0):
    """Projective transformation (homography).

    Perform a projective transformation (homography) of a floating
    point image (single or double precision), using interpolation.

    For each pixel, given its homogeneous coordinate :math:`\mathbf{x}
    = [x, y, 1]^T`, its target position is calculated by multiplying
    with the given matrix, :math:`H`, to give :math:`H \mathbf{x}`.
    E.g., to rotate by theta degrees clockwise, the matrix should be::

      [[cos(theta) -sin(theta) 0]
       [sin(theta)  cos(theta) 0]
       [0            0         1]]

    or, to translate x by 10 and y by 20::

      [[1 0 10]
       [0 1 20]
       [0 0 1 ]].

    Parameters
    ----------
    image : 2-D array
        Input image.
    H : array of shape ``(3, 3)``
        Transformation matrix H that defines the homography.
    output_shape : tuple (rows, cols), optional
        Shape of the output image generated (default None).
    order : {0, 1, 2, 3}, optional
        Order of interpolation::
        * 0: Nearest-neighbor
        * 1: Bi-linear (default)
        * 2: Bi-quadratic
        * 3: Bi-cubic
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : string, optional (default 0)
        Used in conjunction with mode 'C' (constant), the value
        outside the image boundaries.

    Notes
    -----
    Modes 'reflect' and 'symmetric' are similar, but differ in whether the edge
    pixels are duplicated during the reflection.  As an example, if an array
    has values [0, 1, 2] and was padded to the right by four values using
    symmetric, the result would be [0, 1, 2, 2, 1, 0, 0], while for reflect it
    would be [0, 1, 2, 1, 0, 1, 2].

    """

    cdef np_floats[:, ::1] img = np.ascontiguousarray(image)
    cdef np_floats[:, ::1] M = np.ascontiguousarray(H)

    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    if mode not in ('constant', 'wrap', 'symmetric', 'reflect', 'edge'):
        raise ValueError("Invalid mode specified.  Please use `constant`, "
                         "`edge`, `wrap`, `reflect` or `symmetric`.")
    cdef char mode_c = ord(mode[0].upper())

    cdef Py_ssize_t out_r, out_c
    if output_shape is None:
        out_r = int(img.shape[0])
        out_c = int(img.shape[1])
    else:
        out_r = int(output_shape[0])
        out_c = int(output_shape[1])

    cdef np_floats[:, ::1] out = np.zeros((out_r, out_c), dtype=dtype)

    cdef Py_ssize_t tfr, tfc
    cdef np_floats r, c
    cdef Py_ssize_t rows = img.shape[0]
    cdef Py_ssize_t cols = img.shape[1]

    cdef void (*transform_func)(np_floats, np_floats, np_floats*,
                                np_floats*, np_floats*) nogil
    if M[2, 0] == 0 and M[2, 1] == 0 and M[2, 2] == 1:
        if M[0, 1] == 0 and M[1, 0] == 0:
            transform_func = _transform_metric
        else:
            transform_func = _transform_affine
    else:
        transform_func = _transform_projective

    cdef void (*interp_func)(np_floats*, Py_ssize_t , Py_ssize_t ,
                             np_floats, np_floats, char, np_floats,
                             np_floats*) nogil
    if order == 0:
        interp_func = nearest_neighbour_interpolation[np_floats, np_floats,
                                                      np_floats]
    elif order == 1:
        interp_func = bilinear_interpolation[np_floats, np_floats, np_floats]
    elif order == 2:
        interp_func = biquadratic_interpolation[np_floats, np_floats, np_floats]
    elif order == 3:
        interp_func = bicubic_interpolation[np_floats, np_floats, np_floats]
    else:
        raise ValueError("Unsupported interpolation order", order)

    with nogil:
        for tfr in range(out_r):
            for tfc in range(out_c):
                transform_func(tfc, tfr, &M[0, 0], &c, &r)
                interp_func(&img[0, 0], rows, cols, r, c,
                            mode_c, cval, &out[tfr, tfc])

    return np.asarray(out)


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


def warp(image, inverse_map, map_args={}, output_shape=None, order=None,
         mode='constant', cval=0., clip=True, preserve_range=False):
    """Warp an image according to a given coordinate transformation.

    Parameters
    ----------
    image : ndarray
        Input image.
    inverse_map : transformation object, callable ``cr = f(cr, **kwargs)``, or ndarray
        Inverse coordinate map, which transforms coordinates in the output
        images into their corresponding coordinates in the input image.

        There are a number of different options to define this map, depending
        on the dimensionality of the input image. A 2-D image can have 2
        dimensions for gray-scale images, or 3 dimensions with color
        information.

         - For 2-D images, you can directly pass a transformation object,
           e.g. `skimage.transform.SimilarityTransform`, or its inverse.
         - For 2-D images, you can pass a ``(3, 3)`` homogeneous
           transformation matrix, e.g.
           `skimage.transform.SimilarityTransform.params`.
         - For 2-D images, a function that transforms a ``(M, 2)`` array of
           ``(col, row)`` coordinates in the output image to their
           corresponding coordinates in the input image. Extra parameters to
           the function can be specified through `map_args`.
         - For N-D images, you can directly pass an array of coordinates.
           The first dimension specifies the coordinates in the input image,
           while the subsequent dimensions determine the position in the
           output image. E.g. in case of 2-D images, you need to pass an array
           of shape ``(2, rows, cols)``, where `rows` and `cols` determine the
           shape of the output image, and the first dimension contains the
           ``(row, col)`` coordinate in the input image.
           See `scipy.ndimage.map_coordinates` for further documentation.

        Note, that a ``(3, 3)`` matrix is interpreted as a homogeneous
        transformation matrix, so you cannot interpolate values from a 3-D
        input, if the output is of shape ``(3,)``.

        See example section for usage.
    map_args : dict, optional
        Keyword arguments passed to `inverse_map`.
    output_shape : tuple (rows, cols), optional
        Shape of the output image generated. By default the shape of the input
        image is preserved.  Note that, even for multi-band images, only rows
        and columns need to be specified.
    order : int, optional
        The order of interpolation. The order has to be in the range 0-5:
         - 0: Nearest-neighbor
         - 1: Bi-linear (default)
         - 2: Bi-quadratic
         - 3: Bi-cubic
         - 4: Bi-quartic
         - 5: Bi-quintic

         Default is 0 if image.dtype is bool and 1 otherwise.
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
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html

    Returns
    -------
    warped : double ndarray
        The warped input image.

    Notes
    -----
    - The input image is converted to a `double` image.
    - In case of a `SimilarityTransform`, `AffineTransform` and
      `ProjectiveTransform` and `order` in [0, 3] this function uses the
      underlying transformation matrix to warp the image with a much faster
      routine.

    Examples
    --------
    >>> from skimage.transform import warp
    >>> from skimage import data
    >>> image = data.camera()

    The following image warps are all equal but differ substantially in
    execution time. The image is shifted to the bottom.

    Use a geometric transform to warp an image (fast):

    >>> from skimage.transform import SimilarityTransform
    >>> tform = SimilarityTransform(translation=(0, -10))
    >>> warped = warp(image, tform)

    Use a callable (slow):

    >>> def shift_down(xy):
    ...     xy[:, 1] -= 10
    ...     return xy
    >>> warped = warp(image, shift_down)

    Use a transformation matrix to warp an image (fast):

    >>> matrix = np.array([[1, 0, 0], [0, 1, -10], [0, 0, 1]])
    >>> warped = warp(image, matrix)
    >>> from skimage.transform import ProjectiveTransform
    >>> warped = warp(image, ProjectiveTransform(matrix=matrix))

    You can also use the inverse of a geometric transformation (fast):

    >>> warped = warp(image, tform.inverse)

    For N-D images you can pass a coordinate array, that specifies the
    coordinates in the input image for every element in the output image. E.g.
    if you want to rescale a 3-D cube, you can do:

    >>> cube_shape = np.array([30, 30, 30])
    >>> rng = np.random.default_rng()
    >>> cube = rng.random(cube_shape)

    Setup the coordinate array, that defines the scaling:

    >>> scale = 0.1
    >>> output_shape = (scale * cube_shape).astype(int)
    >>> coords0, coords1, coords2 = np.mgrid[:output_shape[0],
    ...                    :output_shape[1], :output_shape[2]]
    >>> coords = np.array([coords0, coords1, coords2])

    Assume that the cube contains spatial data, where the first array element
    center is at coordinate (0.5, 0.5, 0.5) in real space, i.e. we have to
    account for this extra offset when scaling the image:

    >>> coords = (coords + 0.5) / scale - 0.5
    >>> warped = warp(cube, coords)

    """

    if image.size == 0:
        raise ValueError("Cannot warp empty image with dimensions",
                         image.shape)

    order = _validate_interpolation_order(image.dtype, order)

    if order > 0:
        image = convert_to_float(image, preserve_range)
        if image.dtype == np.float16:
            image = image.astype(np.float32)

    input_shape = np.array(image.shape)

    if output_shape is None:
        output_shape = input_shape
    else:
        output_shape = safe_as_int(output_shape)

    warped = None

    if order == 2:
        # When fixing this issue, make sure to fix the branches further
        # below in this function
        warn("Bi-quadratic interpolation behavior has changed due "
             "to a bug in the implementation of scikit-image. "
             "The new version now serves as a wrapper "
             "around SciPy's interpolation functions, which itself "
             "is not verified to be a correct implementation. Until "
             "skimage's implementation is fixed, we recommend "
             "to use bi-linear or bi-cubic interpolation instead.")

    if order in (1, 3) and not map_args:
        # use fast Cython version for specific interpolation orders and input

        matrix = None

        if isinstance(inverse_map, np.ndarray) and inverse_map.shape == (3, 3):
            # inverse_map is a transformation matrix as numpy array
            matrix = inverse_map

        elif isinstance(inverse_map, HOMOGRAPHY_TRANSFORMS):
            # inverse_map is a homography
            matrix = inverse_map.params

        elif (hasattr(inverse_map, '__name__') and
              inverse_map.__name__ == 'inverse' and
              get_bound_method_class(inverse_map) in HOMOGRAPHY_TRANSFORMS):
            # inverse_map is the inverse of a homography
            matrix = np.linalg.inv(inverse_map.__self__.params)

        if matrix is not None:
            matrix = matrix.astype(image.dtype)
            ctype = 'float32_t' if image.dtype == np.float32 else 'float64_t'
            if image.ndim == 2:
                warped = _warp_fast[ctype](image, matrix,
                                           output_shape=output_shape,
                                           order=order, mode=mode, cval=cval)
            elif image.ndim == 3:
                dims = []
                for dim in range(image.shape[2]):
                    dims.append(_warp_fast[ctype](image[..., dim], matrix,
                                                  output_shape=output_shape,
                                                  order=order, mode=mode,
                                                  cval=cval))
                warped = np.dstack(dims)

    if warped is None:
        # use ndi.map_coordinates

        if (isinstance(inverse_map, np.ndarray) and
                inverse_map.shape == (3, 3)):
            # inverse_map is a transformation matrix as numpy array,
            # this is only used for order >= 4.
            inverse_map = ProjectiveTransform(matrix=inverse_map)

        if isinstance(inverse_map, np.ndarray):
            # inverse_map is directly given as coordinates
            coords = inverse_map
        else:
            # inverse_map is given as function, that transforms (N, 2)
            # destination coordinates to their corresponding source
            # coordinates. This is only supported for 2(+1)-D images.

            if image.ndim < 2 or image.ndim > 3:
                raise ValueError("Only 2-D images (grayscale or color) are "
                                 "supported, when providing a callable "
                                 "`inverse_map`.")

            def coord_map(*args):
                return inverse_map(*args, **map_args)

            if len(input_shape) == 3 and len(output_shape) == 2:
                # Input image is 2D and has color channel, but output_shape is
                # given for 2-D images. Automatically add the color channel
                # dimensionality.
                output_shape = (output_shape[0], output_shape[1],
                                input_shape[2])

            coords = warp_coords(coord_map, output_shape)

        # Pre-filtering not necessary for order 0, 1 interpolation
        prefilter = order > 1

        ndi_mode = _to_ndimage_mode(mode)
        warped = ndi.map_coordinates(image, coords, prefilter=prefilter,
                                     mode=ndi_mode, order=order, cval=cval)

    _clip_warp_output(image, warped, order, mode, cval, clip)

    return warped
