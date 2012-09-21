import numpy as np
from ._geometric import (warp, SimilarityTransform, AffineTransform,
                         ProjectiveTransform)


def resize(image, output_shape, order=1, mode='constant', cval=0.):
    """Resize image.

    Parameters
    ----------
    image : ndarray
        Input image.
    output_shape : tuple or ndarray
        Size of the generated output image `(rows, cols)`.

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

    rows, cols = output_shape
    orig_rows, orig_cols = image.shape[0], image.shape[1]

    rscale = float(orig_rows) / rows
    cscale = float(orig_cols) / cols

    # 3 control points necessary to estimate exact AffineTransform
    src_corners = np.array([[1, 1], [1, rows], [cols, rows]]) - 1
    dst_corners = np.zeros(src_corners.shape, dtype=np.double)
    # take into account that 0th pixel is at position (0.5, 0.5)
    dst_corners[:, 0] = cscale * (src_corners[:, 0] + 0.5) - 0.5
    dst_corners[:, 1] = rscale * (src_corners[:, 1] + 0.5) - 0.5

    tform = AffineTransform()
    tform.estimate(src_corners, dst_corners)

    return warp(image, tform, output_shape=output_shape, order=order,
                mode=mode, cval=cval)


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
    if not resize:
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


def homography(image, H, output_shape=None, order=1,
               mode='constant', cval=0.):
    """
    .. note:: Deprecated in skimage 0.7
        `homography` will be removed in skimage 0.8, it is replaced by
        `warp` because the latter provides the same functionality::

            warp(image, ProjectiveTransform(H))

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
