import numpy as np
from scipy import ndimage

from ..measure import block_reduce
from ._geometric import (warp, SimilarityTransform, AffineTransform,
                         _convert_warp_input, _clip_warp_output)


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False):
    """Resize image to match a certain size.

    Performs interpolation to up-size or down-size images. For down-sampling
    N-dimensional images by applying the arithmetic sum or mean, see
    `skimage.measure.local_sum` and `skimage.transform.downscale_local_mean`,
    respectively.

    Parameters
    ----------
    image : ndarray
        Input image.
    output_shape : tuple or ndarray
        Size of the generated output image `(rows, cols[, dim])`. If `dim` is
        not provided, the number of channels is preserved. In case the number
        of input channels does not equal the number of output channels a
        3-dimensional interpolation is applied.

    Returns
    -------
    resized : ndarray
        Resized version of the input.

    Other parameters
    ----------------
    order : int, optional
        The order of the spline interpolation, default is 1. The order has to
        be in the range 0-5. See `skimage.transform.warp` for detail.
    mode : string, optional
        Points outside the boundaries of the input are filled according
        to the given mode ('constant', 'nearest', 'reflect' or 'wrap').
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

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.transform import resize
    >>> image = data.camera()
    >>> resize(image, (100, 100)).shape
    (100, 100)

    """

    rows, cols = output_shape[0], output_shape[1]
    orig_rows, orig_cols = image.shape[0], image.shape[1]

    row_scale = float(orig_rows) / rows
    col_scale = float(orig_cols) / cols

    # 3-dimensional interpolation
    if len(output_shape) == 3 and (image.ndim == 2
                                   or output_shape[2] != image.shape[2]):
        dim = output_shape[2]
        orig_dim = 1 if image.ndim == 2 else image.shape[2]
        dim_scale = float(orig_dim) / dim

        map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
        map_rows = row_scale * (map_rows + 0.5) - 0.5
        map_cols = col_scale * (map_cols + 0.5) - 0.5
        map_dims = dim_scale * (map_dims + 0.5) - 0.5

        coord_map = np.array([map_rows, map_cols, map_dims])

        image = _convert_warp_input(image, preserve_range)

        out = ndimage.map_coordinates(image, coord_map, order=order,
                                      mode=mode, cval=cval)

        _clip_warp_output(image, out, order, mode, cval, clip)

    else:  # 2-dimensional interpolation

        if rows == 1 and cols == 1:
            tform = AffineTransform(translation=(orig_cols / 2.0 - 0.5,
                                                 orig_rows / 2.0 - 0.5))
        else:
            # 3 control points necessary to estimate exact AffineTransform
            src_corners = np.array([[1, 1], [1, rows], [cols, rows]]) - 1
            dst_corners = np.zeros(src_corners.shape, dtype=np.double)
            # take into account that 0th pixel is at position (0.5, 0.5)
            dst_corners[:, 0] = col_scale * (src_corners[:, 0] + 0.5) - 0.5
            dst_corners[:, 1] = row_scale * (src_corners[:, 1] + 0.5) - 0.5

            tform = AffineTransform()
            tform.estimate(src_corners, dst_corners)

        out = warp(image, tform, output_shape=output_shape, order=order,
                   mode=mode, cval=cval, clip=clip,
                   preserve_range=preserve_range)

    return out


def rescale(image, scale, order=1, mode='constant', cval=0, clip=True,
            preserve_range=False):
    """Scale image by a certain factor.

    Performs interpolation to upscale or down-scale images. For down-sampling
    N-dimensional images with integer factors by applying the arithmetic sum or
    mean, see `skimage.measure.local_sum` and
    `skimage.transform.downscale_local_mean`, respectively.

    Parameters
    ----------
    image : ndarray
        Input image.
    scale : {float, tuple of floats}
        Scale factors. Separate scale factors can be defined as
        `(row_scale, col_scale)`.

    Returns
    -------
    scaled : ndarray
        Scaled version of the input.

    Other parameters
    ----------------
    order : int, optional
        The order of the spline interpolation, default is 1. The order has to
        be in the range 0-5. See `skimage.transform.warp` for detail.
    mode : string, optional
        Points outside the boundaries of the input are filled according
        to the given mode ('constant', 'nearest', 'reflect' or 'wrap').
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

    try:
        row_scale, col_scale = scale
    except TypeError:
        row_scale = col_scale = scale

    orig_rows, orig_cols = image.shape[0], image.shape[1]
    rows = np.round(row_scale * orig_rows)
    cols = np.round(col_scale * orig_cols)
    output_shape = (rows, cols)

    return resize(image, output_shape, order=order, mode=mode, cval=cval,
                  clip=clip, preserve_range=preserve_range)


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
        its center, i.e. ``center=(rows / 2 - 0.5, cols / 2 - 0.5)``.

    Returns
    -------
    rotated : ndarray
        Rotated version of the input.

    Other parameters
    ----------------
    order : int, optional
        The order of the spline interpolation, default is 1. The order has to
        be in the range 0-5. See `skimage.transform.warp` for detail.
    mode : string, optional
        Points outside the boundaries of the input are filled according
        to the given mode ('constant', 'nearest', 'reflect' or 'wrap').
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
    tform1 = SimilarityTransform(translation=-center)
    tform2 = SimilarityTransform(rotation=np.deg2rad(angle))
    tform3 = SimilarityTransform(translation=center)
    tform = tform1 + tform2 + tform3

    output_shape = None
    if resize:
        # determine shape of output image
        corners = np.array([[1, 1], [1, rows], [cols, rows], [cols, 1]])
        corners = tform(corners - 1)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_rows = maxr - minr + 1
        out_cols = maxc - minc + 1
        output_shape = np.ceil((out_rows, out_cols))

        # fit output image in new shape
        translation = ((cols - out_cols) / 2., (rows - out_rows) / 2.)
        tform4 = SimilarityTransform(translation=translation)
        tform = tform4 + tform

    return warp(image, tform, output_shape=output_shape, order=order,
                mode=mode, cval=cval, clip=clip, preserve_range=preserve_range)


def downscale_local_mean(image, factors, cval=0, clip=True):
    """Down-sample N-dimensional image by local averaging.

    The image is padded with `cval` if it is not perfectly divisible by the
    integer factors.

    In contrast to the 2-D interpolation in `skimage.transform.resize` and
    `skimage.transform.rescale` this function may be applied to N-dimensional
    images and calculates the local mean of elements in each block of size
    `factors` in the input image.

    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    factors : array_like
        Array containing down-sampling integer factor along each axis.
    cval : float, optional
        Constant padding value if image is not perfectly divisible by the
        integer factors.

    Returns
    -------
    image : ndarray
        Down-sampled image with same number of dimensions as input image.

    Examples
    --------
    >>> a = np.arange(15).reshape(3, 5)
    >>> a
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])
    >>> downscale_local_mean(a, (2, 3))
    array([[ 3.5,  4. ],
           [ 5.5,  4.5]])

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
          output_shape=None, order=1, mode='constant', cval=0, clip=True,
          preserve_range=False):
    """Perform a swirl transformation.

    Parameters
    ----------
    image : ndarray
        Input image.
    center : (row, column) tuple or (2,) ndarray, optional
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
    mode : string, optional
        Points outside the boundaries of the input are filled according
        to the given mode ('constant', 'nearest', 'reflect' or 'wrap').
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

    """

    if center is None:
        center = np.array(image.shape)[:2] / 2

    warp_args = {'center': center,
                 'rotation': rotation,
                 'strength': strength,
                 'radius': radius}

    return warp(image, _swirl_mapping, map_args=warp_args,
                output_shape=output_shape, order=order, mode=mode, cval=cval,
                clip=clip, preserve_range=preserve_range)
