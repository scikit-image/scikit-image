import numpy as np
from scipy import ndimage
from ._geometric import (warp, SimilarityTransform, AffineTransform,
                         ProjectiveTransform)


def resize(image, output_shape, order=1, mode='constant', cval=0.):
    """Resize image to match a certain size.

    Parameters
    ----------
    image : ndarray
        Input image.
    output_shape : tuple or ndarray
        Size of the generated output image `(rows, cols[, dim])`. If `dim` is
        not provided, the number of channels are preserved. In case the number
        of input channels does not equal the number of output channels a
        3-dimensional interpolation is applied.

    Returns
    -------
    resized : ndarray
        Resized version of the input.

    Other parameters
    ----------------
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

        out = ndimage.map_coordinates(image, coord_map, order=order, mode=mode,
                                      cval=cval)

    else: # 2-dimensional interpolation

        # 3 control points necessary to estimate exact AffineTransform
        src_corners = np.array([[1, 1], [1, rows], [cols, rows]]) - 1
        dst_corners = np.zeros(src_corners.shape, dtype=np.double)
        # take into account that 0th pixel is at position (0.5, 0.5)
        dst_corners[:, 0] = col_scale * (src_corners[:, 0] + 0.5) - 0.5
        dst_corners[:, 1] = row_scale * (src_corners[:, 1] + 0.5) - 0.5

        tform = AffineTransform()
        tform.estimate(src_corners, dst_corners)

        out = warp(image, tform, output_shape=output_shape, order=order,
                   mode=mode, cval=cval)

    return out


def rescale(image, scale, order=1, mode='constant', cval=0.):
    """Scale image by a certain factor.

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

    try:
        row_scale, col_scale = scale
    except TypeError:
        row_scale = col_scale = scale

    orig_rows, orig_cols = image.shape[0], image.shape[1]
    rows = np.round(row_scale * orig_rows)
    cols = np.round(col_scale * orig_cols)
    output_shape = (rows, cols)

    return resize(image, output_shape, order=order, mode=mode, cval=cval)


def rotate(image, angle, resize=False, order=1, mode='constant', cval=0.):
    """Rotate image by a certain angle around its center.

    Parameters
    ----------
    image : ndarray
        Input image.
    angle : float
        Rotation angle in degrees in counter-clockwise direction.
    resize: bool, optional
        Determine whether the shape of the output image will be automatically
        calculated, so the complete rotated image exactly fits. Default is
        False.

    Returns
    -------
    rotated : ndarray
        Rotated version of the input.

    Other parameters
    ----------------
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

    rows, cols = image.shape[0], image.shape[1]

    # rotation around center
    translation = np.array((cols, rows)) / 2. - 0.5
    tform1 = SimilarityTransform(translation=-translation)
    tform2 = SimilarityTransform(rotation=np.deg2rad(angle))
    tform3 = SimilarityTransform(translation=translation)
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
                mode=mode, cval=cval)


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
