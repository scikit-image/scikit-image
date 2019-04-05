import numpy as np
from scipy import ndimage as ndi

from ._geometric import (SimilarityTransform, AffineTransform,
                         ProjectiveTransform, _to_ndimage_mode)
from ._warps_cy import _warp_fast

from .._shared.utils import (get_bound_method_class, safe_as_int, warn,
                             convert_to_float)

from .._transform_common import _clip_warp_output, warp_coords


HOMOGRAPHY_TRANSFORMS = (
    SimilarityTransform,
    AffineTransform,
    ProjectiveTransform
)


def rotate(image, angle, resize=False, center=None, order=1, mode='constant',
           cval=0, clip=True, preserve_range=False):
    """Rotate image by a certain angle around its center.

    Parameters
    ----------
    image : ndarray
        Input image.
    angle : float
        Rotation angle in degrees in counter-clockwise direction.
    resize : bool, optional
        Determine whether the shape of the output image will be automatically
        calculated, so the complete rotated image exactly fits. Default is
        False.
    center : iterable of length 2
        The rotation center. If ``center=None``, the image is rotated around
        its center, i.e. ``center=(cols / 2 - 0.5, rows / 2 - 0.5)``.  Please
        note that this parameter is (cols, rows), contrary to normal skimage
        ordering.

    Returns
    -------
    rotated : ndarray
        Rotated version of the input.

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
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html

    Notes
    -----
    Modes 'reflect' and 'symmetric' are similar, but differ in whether the edge
    pixels are duplicated during the reflection.  As an example, if an array
    has values [0, 1, 2] and was padded to the right by four values using
    symmetric, the result would be [0, 1, 2, 2, 1, 0, 0], while for reflect it
    would be [0, 1, 2, 1, 0, 1, 2].

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.transform import rotate
    >>> image = data.camera()
    >>> rotate(image, 2).shape
    (512, 512)
    >>> rotate(image, 2, resize=True).shape
    (530, 530)
    >>> rotate(image, 90, resize=True).shape
    (512, 512)

    """

    rows, cols = image.shape[0], image.shape[1]

    # rotation around center
    if center is None:
        center = np.array((cols, rows)) / 2. - 0.5
    else:
        center = np.asarray(center)
    tform1 = SimilarityTransform(translation=center)
    tform2 = SimilarityTransform(rotation=np.deg2rad(angle))
    tform3 = SimilarityTransform(translation=-center)
    tform = tform3 + tform2 + tform1

    output_shape = None
    if resize:
        # determine shape of output image
        corners = np.array([
            [0, 0],
            [0, rows - 1],
            [cols - 1, rows - 1],
            [cols - 1, 0]
        ])
        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_rows = maxr - minr + 1
        out_cols = maxc - minc + 1
        output_shape = np.ceil((out_rows, out_cols))

        # fit output image in new shape
        translation = (minc, minr)
        tform4 = SimilarityTransform(translation=translation)
        tform = tform4 + tform

    # Make sure the transform is exactly affine, to ensure fast warping.
    tform.params[2] = (0, 0, 1)

    return warp(image, tform, output_shape=output_shape, order=order,
                mode=mode, cval=cval, clip=clip, preserve_range=preserve_range)


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
          output_shape=None, order=1, mode='reflect', cval=0, clip=True,
          preserve_range=False):
    """Perform a swirl transformation.

    Parameters
    ----------
    image : ndarray
        Input image.
    center : (column, row) tuple or (2,) ndarray, optional
        Center coordinate of transformation.
    strength : float, optional
        The amount of swirling applied.
    radius : float, optional
        The extent of the swirl in pixels.  The effect dies out
        rapidly beyond `radius`.
    rotation : float, optional
        Additional rotation applied to the image.

    Returns
    -------
    swirled : ndarray
        Swirled version of the input.

    Other parameters
    ----------------
    output_shape : tuple (rows, cols), optional
        Shape of the output image generated. By default the shape of the input
        image is preserved.
    order : int, optional
        The order of the spline interpolation, default is 1. The order has to
        be in the range 0-5. See `skimage.transform.warp` for detail.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode, with 'constant' used as the default. Modes match
        the behaviour of `numpy.pad`.
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

    """
    if center is None:
        center = np.array(image.shape)[:2][::-1] / 2

    warp_args = {'center': center,
                 'rotation': rotation,
                 'strength': strength,
                 'radius': radius}

    return warp(image, _swirl_mapping, map_args=warp_args,
                output_shape=output_shape, order=order, mode=mode, cval=cval,
                clip=clip, preserve_range=preserve_range)


def warp(image, inverse_map, map_args={}, output_shape=None, order=1,
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
    >>> cube = np.random.rand(*cube_shape)

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
        raise ValueError("Cannot warp empty image with dimensions", image.shape)

    image = convert_to_float(image, preserve_range)

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

    if order in (0, 1, 3) and not map_args:
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
            matrix = matrix.astype(np.double)
            if image.ndim == 2:
                warped = _warp_fast(image, matrix,
                                    output_shape=output_shape,
                                    order=order, mode=mode, cval=cval)
            elif image.ndim == 3:
                dims = []
                for dim in range(image.shape[2]):
                    dims.append(_warp_fast(image[..., dim], matrix,
                                           output_shape=output_shape,
                                           order=order, mode=mode, cval=cval))
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
