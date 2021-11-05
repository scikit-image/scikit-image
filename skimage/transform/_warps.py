import numpy as np
from numpy.lib import NumpyVersion
import scipy
from scipy import ndimage as ndi

from .._shared._geometric import (SimilarityTransform, AffineTransform)
from .._shared.warp import warp, warp_coords, _clip_warp_output
from .._shared.block import block_reduce

from .._shared.utils import (safe_as_int, warn,
                             convert_to_float, _to_ndimage_mode,
                             _validate_interpolation_order,
                             channel_as_last_axis,
                             deprecate_multichannel_kwarg)


def _preprocess_resize_output_shape(image, output_shape):
    """Validate resize output shape according to input image.

    Parameters
    ----------
    image: ndarray
        Image to be resized.
    output_shape: tuple or ndarray
        Size of the generated output image `(rows, cols[, ...][, dim])`. If
        `dim` is not provided, the number of channels is preserved.

    Returns
    -------
    image: ndarray
        The input image, but with additional singleton dimensions appended in
        the case where ``len(output_shape) > input.ndim``.
    output_shape: tuple
        The output image converted to tuple.

    Raises
    ------
    ValueError:
        If output_shape length is smaller than the image number of
        dimensions

    Notes
    -----
    The input image is reshaped if its number of dimensions is not
    equal to output_shape_length.

    """
    output_ndim = len(output_shape)
    input_shape = image.shape
    if output_ndim > image.ndim:
        # append dimensions to input_shape
        input_shape += (1, ) * (output_ndim - image.ndim)
        image = np.reshape(image, input_shape)
    elif output_ndim == image.ndim - 1:
        # multichannel case: append shape of last axis
        output_shape = output_shape + (image.shape[-1], )
    elif output_ndim < image.ndim:
        raise ValueError("output_shape length cannot be smaller than the "
                         "image number of dimensions")

    return image, output_shape


def resize(image, output_shape, order=None, mode='reflect', cval=0, clip=True,
           preserve_range=False, anti_aliasing=None, anti_aliasing_sigma=None):
    """Resize image to match a certain size.

    Performs interpolation to up-size or down-size N-dimensional images. Note
    that anti-aliasing should be enabled when down-sizing images to avoid
    aliasing artifacts. For down-sampling with an integer factor also see
    `skimage.transform.downscale_local_mean`.

    Parameters
    ----------
    image : ndarray
        Input image.
    output_shape : tuple or ndarray
        Size of the generated output image `(rows, cols[, ...][, dim])`. If
        `dim` is not provided, the number of channels is preserved. In case the
        number of input channels does not equal the number of output channels a
        n-dimensional interpolation is applied.

    Returns
    -------
    resized : ndarray
        Resized version of the input.

    Other parameters
    ----------------
    order : int, optional
        The order of the spline interpolation, default is 0 if
        image.dtype is bool and 1 otherwise. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.
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
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    anti_aliasing : bool, optional
        Whether to apply a Gaussian filter to smooth the image prior
        to down-scaling. It is crucial to filter when down-sampling
        the image to avoid aliasing artifacts. If input image data
        type is bool, no anti-aliasing is applied.
    anti_aliasing_sigma : {float, tuple of floats}, optional
        Standard deviation for Gaussian filtering to avoid aliasing artifacts.
        By default, this value is chosen as (s - 1) / 2 where s is the
        down-scaling factor, where s > 1. For the up-size case, s < 1, no
        anti-aliasing is performed prior to rescaling.

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
    >>> from skimage.transform import resize
    >>> image = data.camera()
    >>> resize(image, (100, 100)).shape
    (100, 100)

    """

    image, output_shape = _preprocess_resize_output_shape(image, output_shape)
    input_shape = image.shape

    if image.dtype == np.float16:
        image = image.astype(np.float32)

    if anti_aliasing is None:
        anti_aliasing = not image.dtype == bool

    if image.dtype == bool and anti_aliasing:
        raise ValueError("anti_aliasing must be False for boolean images")

    factors = (np.asarray(input_shape, dtype=float) /
               np.asarray(output_shape, dtype=float))

    # Translate modes used by np.pad to those used by scipy.ndimage
    ndi_mode = _to_ndimage_mode(mode)
    if anti_aliasing:
        if anti_aliasing_sigma is None:
            anti_aliasing_sigma = np.maximum(0, (factors - 1) / 2)
        else:
            anti_aliasing_sigma = \
                np.atleast_1d(anti_aliasing_sigma) * np.ones_like(factors)
            if np.any(anti_aliasing_sigma < 0):
                raise ValueError("Anti-aliasing standard deviation must be "
                                 "greater than or equal to zero")
            elif np.any((anti_aliasing_sigma > 0) & (factors <= 1)):
                warn("Anti-aliasing standard deviation greater than zero but "
                     "not down-sampling along all axes")
        image = ndi.gaussian_filter(image, anti_aliasing_sigma,
                                    cval=cval, mode=ndi_mode)

    if NumpyVersion(scipy.__version__) >= '1.6.0':
        # The grid_mode kwarg was introduced in SciPy 1.6.0
        order = _validate_interpolation_order(image.dtype, order)
        zoom_factors = [1 / f for f in factors]
        if order > 0:
            image = convert_to_float(image, preserve_range)
        out = ndi.zoom(image, zoom_factors, order=order, mode=ndi_mode,
                       cval=cval, grid_mode=True)
        _clip_warp_output(image, out, order, mode, cval, clip)

    # TODO: Remove the fallback code below once SciPy >= 1.6.0 is required.

    # 2-dimensional interpolation
    elif len(output_shape) == 2 or (len(output_shape) == 3 and
                                    output_shape[2] == input_shape[2]):
        rows = output_shape[0]
        cols = output_shape[1]
        input_rows = input_shape[0]
        input_cols = input_shape[1]
        if rows == 1 and cols == 1:
            tform = AffineTransform(translation=(input_cols / 2.0 - 0.5,
                                                 input_rows / 2.0 - 0.5))
        else:
            # 3 control points necessary to estimate exact AffineTransform
            src_corners = np.array([[1, 1], [1, rows], [cols, rows]]) - 1
            dst_corners = np.zeros(src_corners.shape, dtype=np.double)
            # take into account that 0th pixel is at position (0.5, 0.5)
            dst_corners[:, 0] = factors[1] * (src_corners[:, 0] + 0.5) - 0.5
            dst_corners[:, 1] = factors[0] * (src_corners[:, 1] + 0.5) - 0.5

            tform = AffineTransform()
            tform.estimate(src_corners, dst_corners)

        # Make sure the transform is exactly metric, to ensure fast warping.
        tform.params[2] = (0, 0, 1)
        tform.params[0, 1] = 0
        tform.params[1, 0] = 0

        out = warp(image, tform, output_shape=output_shape, order=order,
                   mode=mode, cval=cval, clip=clip,
                   preserve_range=preserve_range)

    else:  # n-dimensional interpolation
        order = _validate_interpolation_order(image.dtype, order)

        coord_arrays = [factors[i] * (np.arange(d) + 0.5) - 0.5
                        for i, d in enumerate(output_shape)]

        coord_map = np.array(np.meshgrid(*coord_arrays,
                                         sparse=False,
                                         indexing='ij'))

        image = convert_to_float(image, preserve_range)

        out = ndi.map_coordinates(image, coord_map, order=order,
                                  mode=ndi_mode, cval=cval)

        _clip_warp_output(image, out, order, mode, cval, clip)

    return out


@channel_as_last_axis()
@deprecate_multichannel_kwarg(multichannel_position=7)
def rescale(image, scale, order=None, mode='reflect', cval=0, clip=True,
            preserve_range=False, multichannel=False,
            anti_aliasing=None, anti_aliasing_sigma=None, *,
            channel_axis=None):
    """Scale image by a certain factor.

    Performs interpolation to up-scale or down-scale N-dimensional images.
    Note that anti-aliasing should be enabled when down-sizing images to avoid
    aliasing artifacts. For down-sampling with an integer factor also see
    `skimage.transform.downscale_local_mean`.

    Parameters
    ----------
    image : ndarray
        Input image.
    scale : {float, tuple of floats}
        Scale factors. Separate scale factors can be defined as
        `(rows, cols[, ...][, dim])`.

    Returns
    -------
    scaled : ndarray
        Scaled version of the input.

    Other parameters
    ----------------
    order : int, optional
        The order of the spline interpolation, default is 0 if
        image.dtype is bool and 1 otherwise. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.
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
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension. This argument is deprecated:
        specify `channel_axis` instead.
    anti_aliasing : bool, optional
        Whether to apply a Gaussian filter to smooth the image prior
        to down-scaling. It is crucial to filter when down-sampling
        the image to avoid aliasing artifacts. If input image data
        type is bool, no anti-aliasing is applied.
    anti_aliasing_sigma : {float, tuple of floats}, optional
        Standard deviation for Gaussian filtering to avoid aliasing artifacts.
        By default, this value is chosen as (s - 1) / 2 where s is the
        down-scaling factor.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

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
    >>> from skimage.transform import rescale
    >>> image = data.camera()
    >>> rescale(image, 0.1).shape
    (51, 51)
    >>> rescale(image, 0.5).shape
    (256, 256)

    """
    scale = np.atleast_1d(scale)
    multichannel = channel_axis is not None
    if len(scale) > 1:
        if ((not multichannel and len(scale) != image.ndim) or
                (multichannel and len(scale) != image.ndim - 1)):
            raise ValueError("Supply a single scale, or one value per spatial "
                             "axis")
        if multichannel:
            scale = np.concatenate((scale, [1]))
    orig_shape = np.asarray(image.shape)
    output_shape = np.maximum(np.round(scale * orig_shape), 1)
    if multichannel:  # don't scale channel dimension
        output_shape[-1] = orig_shape[-1]

    return resize(image, output_shape, order=order, mode=mode, cval=cval,
                  clip=clip, preserve_range=preserve_range,
                  anti_aliasing=anti_aliasing,
                  anti_aliasing_sigma=anti_aliasing_sigma)


def rotate(image, angle, resize=False, center=None, order=None,
           mode='constant', cval=0, clip=True, preserve_range=False):
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
        The order of the spline interpolation, default is 0 if
        image.dtype is bool and 1 otherwise. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.
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

    if image.dtype == np.float16:
        image = image.astype(np.float32)

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
        output_shape = np.around((out_rows, out_cols))

        # fit output image in new shape
        translation = (minc, minr)
        tform4 = SimilarityTransform(translation=translation)
        tform = tform4 + tform

    # Make sure the transform is exactly affine, to ensure fast warping.
    tform.params[2] = (0, 0, 1)

    return warp(image, tform, output_shape=output_shape, order=order,
                mode=mode, cval=cval, clip=clip, preserve_range=preserve_range)


def downscale_local_mean(image, factors, cval=0, clip=True):
    """Down-sample N-dimensional image by local averaging.

    The image is padded with `cval` if it is not perfectly divisible by the
    integer factors.

    In contrast to interpolation in `skimage.transform.resize` and
    `skimage.transform.rescale` this function calculates the local mean of
    elements in each block of size `factors` in the input image.

    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    factors : array_like
        Array containing down-sampling integer factor along each axis.
    cval : float, optional
        Constant padding value if image is not perfectly divisible by the
        integer factors.
    clip : bool, optional
        Unused, but kept here for API consistency with the other transforms
        in this module. (The local mean will never fall outside the range
        of values in the input image, assuming the provided `cval` also
        falls within that range.)

    Returns
    -------
    image : ndarray
        Down-sampled image with same number of dimensions as input image.
        For integer inputs, the output dtype will be ``float64``.
        See :func:`numpy.mean` for details.

    Examples
    --------
    >>> a = np.arange(15).reshape(3, 5)
    >>> a
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])
    >>> downscale_local_mean(a, (2, 3))
    array([[3.5, 4. ],
           [5.5, 4.5]])

    """
    return block_reduce(image, factors, np.mean, cval)


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
          output_shape=None, order=None, mode='reflect', cval=0, clip=True,
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
        The order of the spline interpolation, default is 0 if
        image.dtype is bool and 1 otherwise. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.
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


def _linear_polar_mapping(output_coords, k_angle, k_radius, center):
    """Inverse mapping function to convert from cartesian to polar coordinates

    Parameters
    ----------
    output_coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the output image
    k_angle : float
        Scaling factor that relates the intended number of rows in the output
        image to angle: ``k_angle = nrows / (2 * np.pi)``
    k_radius : float
        Scaling factor that relates the radius of the circle bounding the
        area to be transformed to the intended number of columns in the output
        image: ``k_radius = ncols / radius``
    center : tuple (row, col)
        Coordinates that represent the center of the circle that bounds the
        area to be transformed in an input image.

    Returns
    -------
    coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the input image that
        correspond to the `output_coords` given as input.
    """
    angle = output_coords[:, 1] / k_angle
    rr = ((output_coords[:, 0] / k_radius) * np.sin(angle)) + center[0]
    cc = ((output_coords[:, 0] / k_radius) * np.cos(angle)) + center[1]
    coords = np.column_stack((cc, rr))
    return coords


def _log_polar_mapping(output_coords, k_angle, k_radius, center):
    """Inverse mapping function to convert from cartesian to polar coordinates

    Parameters
    ----------
    output_coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the output image
    k_angle : float
        Scaling factor that relates the intended number of rows in the output
        image to angle: ``k_angle = nrows / (2 * np.pi)``
    k_radius : float
        Scaling factor that relates the radius of the circle bounding the
        area to be transformed to the intended number of columns in the output
        image: ``k_radius = width / np.log(radius)``
    center : tuple (row, col)
        Coordinates that represent the center of the circle that bounds the
        area to be transformed in an input image.

    Returns
    -------
    coords : ndarray
        `(M, 2)` array of `(col, row)` coordinates in the input image that
        correspond to the `output_coords` given as input.
    """
    angle = output_coords[:, 1] / k_angle
    rr = ((np.exp(output_coords[:, 0] / k_radius)) * np.sin(angle)) + center[0]
    cc = ((np.exp(output_coords[:, 0] / k_radius)) * np.cos(angle)) + center[1]
    coords = np.column_stack((cc, rr))
    return coords


@channel_as_last_axis()
@deprecate_multichannel_kwarg()
def warp_polar(image, center=None, *, radius=None, output_shape=None,
               scaling='linear', multichannel=False, channel_axis=None,
               **kwargs):
    """Remap image to polar or log-polar coordinates space.

    Parameters
    ----------
    image : ndarray
        Input image. Only 2-D arrays are accepted by default. 3-D arrays are
        accepted if a `channel_axis` is specified.
    center : tuple (row, col), optional
        Point in image that represents the center of the transformation (i.e.,
        the origin in cartesian space). Values can be of type `float`.
        If no value is given, the center is assumed to be the center point
        of the image.
    radius : float, optional
        Radius of the circle that bounds the area to be transformed.
    output_shape : tuple (row, col), optional
    scaling : {'linear', 'log'}, optional
        Specify whether the image warp is polar or log-polar. Defaults to
        'linear'.
    multichannel : bool, optional
        Whether the image is a 3-D array in which the third axis is to be
        interpreted as multiple channels. If set to `False` (default), only 2-D
        arrays are accepted. This argument is deprecated: specify
        `channel_axis` instead.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.
    **kwargs : keyword arguments
        Passed to `transform.warp`.

    Returns
    -------
    warped : ndarray
        The polar or log-polar warped image.

    Examples
    --------
    Perform a basic polar warp on a grayscale image:

    >>> from skimage import data
    >>> from skimage.transform import warp_polar
    >>> image = data.checkerboard()
    >>> warped = warp_polar(image)

    Perform a log-polar warp on a grayscale image:

    >>> warped = warp_polar(image, scaling='log')

    Perform a log-polar warp on a grayscale image while specifying center,
    radius, and output shape:

    >>> warped = warp_polar(image, (100,100), radius=100,
    ...                     output_shape=image.shape, scaling='log')

    Perform a log-polar warp on a color image:

    >>> image = data.astronaut()
    >>> warped = warp_polar(image, scaling='log', channel_axis=-1)
    """
    multichannel = channel_axis is not None
    if image.ndim != 2 and not multichannel:
        raise ValueError(f'Input array must be 2-dimensional when '
                         f'`channel_axis=None`, got {image.ndim}')

    if image.ndim != 3 and multichannel:
        raise ValueError(f'Input array must be 3-dimensional when '
                         f'`channel_axis` is specified, got {image.ndim}')

    if center is None:
        center = (np.array(image.shape)[:2] / 2) - 0.5

    if radius is None:
        w, h = np.array(image.shape)[:2] / 2
        radius = np.sqrt(w ** 2 + h ** 2)

    if output_shape is None:
        height = 360
        width = int(np.ceil(radius))
        output_shape = (height, width)
    else:
        output_shape = safe_as_int(output_shape)
        height = output_shape[0]
        width = output_shape[1]

    if scaling == 'linear':
        k_radius = width / radius
        map_func = _linear_polar_mapping
    elif scaling == 'log':
        k_radius = width / np.log(radius)
        map_func = _log_polar_mapping
    else:
        raise ValueError("Scaling value must be in {'linear', 'log'}")

    k_angle = height / (2 * np.pi)
    warp_args = {'k_angle': k_angle, 'k_radius': k_radius, 'center': center}

    warped = warp(image, map_func, map_args=warp_args,
                  output_shape=output_shape, **kwargs)

    return warped


def _local_mean_weights(old_size, new_size, grid_mode, dtype):
    """Create a 2D weight matrix for resizing with the local mean.

    Parameters
    ----------
    old_size: int
        Old size.
    new_size: int
        New size.
    grid_mode : bool
        Whether to use grid data model of pixel/voxel model for
        average weights computation.
    dtype: dtype
        Output array data type.

    Returns
    -------
    weights: (new_size, old_size) array
        Rows sum to 1.

    """
    if grid_mode:
        old_breaks = np.linspace(0, old_size, num=old_size + 1, dtype=dtype)
        new_breaks = np.linspace(0, old_size, num=new_size + 1, dtype=dtype)
    else:
        old, new = old_size - 1, new_size - 1
        old_breaks = np.pad(np.linspace(0.5, old - 0.5, old, dtype=dtype),
                            1, 'constant', constant_values=(0, old))
        if new == 0:
            val = np.inf
        else:
            val = 0.5 * old / new
        new_breaks = np.pad(np.linspace(val, old - val, new, dtype=dtype),
                            1, 'constant', constant_values=(0, old))

    upper = np.minimum(new_breaks[1:, np.newaxis], old_breaks[np.newaxis, 1:])
    lower = np.maximum(new_breaks[:-1, np.newaxis],
                       old_breaks[np.newaxis, :-1])

    weights = np.maximum(upper - lower, 0)
    weights /= weights.sum(axis=1, keepdims=True)

    return weights


def resize_local_mean(image, output_shape, grid_mode=True,
                      preserve_range=False, *, channel_axis=None):
    """Resize an array with the local mean / bilinear scaling.

    Parameters
    ----------
    image : ndarray
        Input image. If this is a multichannel image, the axis corresponding
        to channels should be specified using `channel_axis`
    output_shape : tuple or ndarray
        Size of the generated output image. When `channel_axis` is not None,
        the `channel_axis` should either be omitted from `output_shape` or the
        ``output_shape[channel_axis]`` must match
        ``image.shape[channel_axis]``. If the length of `output_shape` exceeds
        image.ndim, additional singleton dimensions will be appended to the
        input ``image`` as needed.
    grid_mode : bool, optional
        Defines ``image`` pixels position: if True, pixels are assumed to be at
        grid intersections, otherwise at cell centers. As a consequence,
        for example, a 1d signal of length 5 is considered to have length 4
        when `grid_mode` is False, but length 5 when `grid_mode` is True. See
        the following visual illustration:

        .. code-block:: text

                | pixel 1 | pixel 2 | pixel 3 | pixel 4 | pixel 5 |
                     |<-------------------------------------->|
                                        vs.
                |<----------------------------------------------->|

        The starting point of the arrow in the diagram above corresponds to
        coordinate location 0 in each mode.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html

    Returns
    -------
    resized : ndarray
        Resized version of the input.

    See Also
    --------
    resize, downscale_local_mean

    Notes
    -----
    This method is sometimes referred to as "area-based" interpolation or
    "pixel mixing" interpolation [1]_. When `grid_mode` is True, it is
    equivalent to using OpenCV's resize with `INTER_AREA` interpolation mode.
    It is commonly used for image downsizing. If the downsizing factors are
    integers, then `downscale_local_mean` should be preferred instead.

    References
    ----------
    .. [1] http://entropymine.com/imageworsener/pixelmixing/

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.transform import resize_local_mean
    >>> image = data.camera()
    >>> resize_local_mean(image, (100, 100)).shape
    (100, 100)

    """
    if channel_axis is not None:
        if channel_axis < -image.ndim or channel_axis >= image.ndim:
            raise ValueError("invalid channel_axis")

        # move channels to last position
        image = np.moveaxis(image, channel_axis, -1)
        nc = image.shape[-1]

        output_ndim = len(output_shape)
        if output_ndim == image.ndim - 1:
            # insert channels dimension at the end
            output_shape = output_shape + (nc,)
        elif output_ndim == image.ndim:
            if output_shape[channel_axis] != nc:
                raise ValueError(
                    "Cannot reshape along the channel_axis. Use "
                    "channel_axis=None to reshape along all axes."
                )
            # move channels to last position in output_shape
            channel_axis = channel_axis % image.ndim
            output_shape = (
                output_shape[:channel_axis] + output_shape[channel_axis:] +
                (nc,)
            )
        else:
            raise ValueError(
                "len(output_shape) must be image.ndim or (image.ndim - 1) "
                "when a channel_axis is specified."
            )
        resized = image
    else:
        resized, output_shape = _preprocess_resize_output_shape(image,
                                                                output_shape)
    resized = convert_to_float(resized, preserve_range)
    dtype = resized.dtype

    for axis, (old_size, new_size) in enumerate(zip(image.shape,
                                                    output_shape)):
        if old_size == new_size:
            continue
        weights = _local_mean_weights(old_size, new_size, grid_mode, dtype)
        product = np.tensordot(resized, weights, [[axis], [-1]])
        resized = np.moveaxis(product, -1, axis)

    if channel_axis is not None:
        # restore channels to original axis
        resized = np.moveaxis(resized, -1, channel_axis)

    return resized
