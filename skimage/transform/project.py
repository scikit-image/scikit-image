"""Image projection.

"""

import numpy as np
from scipy.ndimage import interpolation as ndii
from ._warp import _stackcopy

__all__ = ['homography']

eps = np.finfo(float).eps


def homography(image, H, output_shape=None, order=1,
               mode='constant', cval=0.):
    """Perform a projective transformation (homography) on an image.

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
    if image.ndim < 2:
        raise ValueError("Input must have more than 1 dimension.")

    image = np.atleast_3d(image)
    ishape = np.array(image.shape)
    bands = ishape[2]

    if output_shape is None:
        output_shape = ishape

    coords = np.empty(np.r_[3, output_shape], dtype=float)

    # TODO: Refactor this method to use transform.warp instead.

    # Construct transformed coordinates
    rows, cols = output_shape[:2]
    rows, cols = np.mgrid[:rows, :cols]
    tf_coords = np.empty(shape=cols.shape,
                         dtype=[('cols', float),
                                ('rows', float),
                                ('z', float)])
    tf_coords['cols'], tf_coords['rows'] = cols, rows
    tf_coords['z'] = 1
    tf_coords = tf_coords.view((float, 3))

    tf_coords = np.dot(tf_coords, np.linalg.inv(H).transpose())
    tf_coords[np.absolute(tf_coords) < eps] = 0.

    # normalize coordinates
    tf_coords[..., :2] /= tf_coords[..., 2, np.newaxis]

    # y-coordinate mapping
    _stackcopy(coords[0, ...], tf_coords[..., 1])

    # x-coordinate mapping
    _stackcopy(coords[1, ...], tf_coords[..., 0])

    # colour-coordinate mapping
    coords[2, ...] = range(bands)

    # Prefilter not necessary for order 1 interpolation
    prefilter = order > 1
    mapped = ndii.map_coordinates(image, coords, prefilter=prefilter,
                                  mode=mode, order=order, cval=cval)

    return mapped.squeeze()
