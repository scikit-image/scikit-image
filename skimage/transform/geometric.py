# coding: utf-8
import math
import numpy as np
from scipy import ndimage
from skimage.util import img_as_float


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
    Color images are stored as an ``MxNx3`` or ``MxNx4`` arrays.

    """
    if a.ndim == 3:
        a[:] = b[:, :, np.newaxis]
    else:
        a[:] = b


def _estimate_similarity(src, dst):
    """Determine parameters of the 2D similarity transformation:
        X = a0*x - b0*y + a1
        Y = b0*x + a0*y + a2
    where the homogeneous transformation matrix is:
        [[a0 -b0  a1]
         [b0  a0  b1]
         [0   0    1]]

    """
    xs = src[:, 0]
    ys = src[:, 1]
    xd = dst[:, 0]
    yd = dst[:, 1]
    rows = src.shape[0]

    #: params: a0, a1, b0, b1
    A = np.zeros((rows * 2, 4))
    A[:rows, 0] = xs
    A[:rows, 2] = - ys
    A[:rows, 1] = 1
    A[rows:, 2] = xs
    A[rows:, 0] = ys
    A[rows:, 3] = 1

    b = np.hstack([xd, yd])

    a0, a1, b0, b1 = np.linalg.lstsq(A, b)[0]
    matrix = np.array([[a0, -b0, a1],
                       [b0,  a0, b1],
                       [ 0,   0,  1]])
    return matrix


def _estimate_affine(src, dst):
    """Determine parameters of the 2D affine transformation:
        X = a0*x + a1*y + a2
        Y = b0*x + b1*y + b2
    where the homogeneous transformation matrix is:
        [[a0  a1  a2]
         [b0  b1  b2]
         [0   0    1]]

    """
    xs = src[:, 0]
    ys = src[:, 1]
    xd = dst[:, 0]
    yd = dst[:, 1]
    rows = src.shape[0]

    #: params: a0, a1, a2, b0, b1, b2
    A = np.zeros((rows * 2, 6))
    A[:rows, 0] = xs
    A[:rows, 1] = ys
    A[:rows, 2] = 1
    A[rows:, 3] = xs
    A[rows:, 4] = ys
    A[rows:, 5] = 1

    b = np.hstack([xd, yd])

    a0, a1, a2, b0, b1, b2 = np.linalg.lstsq(A, b)[0]
    matrix = np.array([[a0, a1, a2],
                       [b0, b1, b2],
                       [0,   0,  1]])
    return matrix


def _estimate_projective(src, dst):
    """Determine transformation matrix of the 2D projective transformation:
        X = (a0 + a1*x + a2*y) / (c0*x + c1*y + 1)
        Y = (b0 + b1*x + b2*y) / (c0*x + c1*y + 1)
    where the homogeneous transformation matrix is:
        [[a0  a1  a2]
         [b0  b1  b2]
         [c0  c1   1]]

    """
    xs = src[:, 0]
    ys = src[:, 1]
    xd = dst[:, 0]
    yd = dst[:, 1]
    rows = src.shape[0]

    #: params: a0, a1, a2, b0, b1, b2, c0, c1
    A = np.zeros((rows * 2, 8))
    A[:rows, 0] = xs
    A[:rows, 1] = ys
    A[:rows, 2] = 1
    A[:rows, 6] = - xd * xs
    A[:rows, 7] = - xd * ys
    A[rows:, 3] = xs
    A[rows:, 4] = ys
    A[rows:, 5] = 1
    A[rows:, 6] = - yd * xs
    A[rows:, 7] = - yd * ys

    b = np.hstack([xd, yd])

    a0, a1, a2, b0, b1, b2, c0, c1 = np.linalg.lstsq(A, b)[0]
    matrix = np.array([[a0, a1, a2],
                       [b0, b1, b2],
                       [c0, c1,  1]])
    return matrix


def _estimate_polynomial(src, dst, order):
    """Determine parameters of 2D polynomial transformation of order n:
        X = sum[j=0:n]( sum[i=0:j]( a_ji * x**(j - i) * y**i ))
        Y = sum[j=0:n]( sum[i=0:j]( b_ji * x**(j - i) * y**i ))

    """
    xs = src[:, 0]
    ys = src[:, 1]
    xd = dst[:, 0]
    yd = dst[:, 1]
    rows = src.shape[0]

    # number of unknown polynomial coefficients
    u = (order + 1) * (order + 2)

    A = np.zeros((rows * 2, u))
    pidx = 0
    for j in xrange(order + 1):
        for i in xrange(j + 1):
            A[:rows, pidx] = xs ** (j - i) * ys ** i
            A[rows:, pidx + u / 2] = xs ** (j - i) * ys ** i
            pidx += 1

    b = np.hstack([xd, yd])

    return np.linalg.lstsq(A, b)[0]


def geometric_transform(coords, matrix):
    """Apply 2D geometric transformation.

    Parameters
        ----------
        ttype : Nx2 array
            x, y coordinates to transform
        matrix : 3x3 array
            homogeneous transformation matrix

    Returns
    -------
    coords : Nx2 array
            transformed coordinates
    """
    x, y = np.transpose(coords)
    src = np.vstack((x, y, np.ones_like(x)))
    dst = np.dot(src.transpose(), matrix.transpose())
    # rescale to homogeneous coordinates
    dst[:, 0] /= dst[:, 2]
    dst[:, 1] /= dst[:, 2]
    return dst[:, :2]


def _transform_polynomial(coords, matrix):
    x = coords[:, 0]
    y = coords[:, 1]
    u = len(matrix)
    # number of coefficients -> u = (order + 1) * (order + 2)
    order = int((- 3 + math.sqrt(9 - 4 * (2 - u))) / 2)
    dst = np.zeros(coords.shape)

    pidx = 0
    for j in xrange(order + 1):
        for i in xrange(j + 1):
            dst[:, 0] += matrix[pidx] * x ** (j - i) * y ** i
            dst[:, 1] += matrix[pidx + u / 2] * x ** (j - i) * y ** i
            pidx += 1

    return dst


class GeometricTransformation(object):

    def __init__(self, ttype, params, transform_func):
        """Create geometric transformation which contains the transformation
        parameters and can perform forward and reverse transformations.

        Parameters
        ----------
        ttype : str
            transformation type - one of 'similarity', 'affine', 'projective',
            'polynomial'
        params : array
            transformation parameters
        transform_func : callable = func(coords, matrix)
            transformation function

        """
        self.ttype = ttype
        self.params = params
        self.transform_func = transform_func

    def forward(self, coords):
        """Apply forward transformation.

        Parameters
        ----------
        coords : Nx2 array
            source coordinates

        Returns
        -------
        coords : Nx2 array
            transformed coordinates

        """
        return self.transform_func(coords, self.params)

    def reverse(self, coords):
        """Apply reverse transformation.

        Parameters
        ----------
        coords : Nx2 array
            source coordinates

        Returns
        -------
        coords : Nx2 array
            transformed coordinates

        """
        if self.ttype == 'polynomial':
            raise Exception(
                'There is no explicit way to do the reverse polynomial '
                'transformation. Instead determine the reverse transformation '
                'parameters by exchanging source and destination coordinates.'
                'Then apply the forward transformation.')
        inv_matrix = np.linalg.inv(self.params)
        return self.transform_func(coords, inv_matrix)


ESTIMATED_TRANSFORMATIONS = {
    'similarity': (_estimate_similarity, geometric_transform),
    'affine': (_estimate_affine, geometric_transform),
    'projective': (_estimate_projective, geometric_transform),
    'polynomial': (_estimate_polynomial, _transform_polynomial),
}


def estimate_transformation(ttype, *args, **kwargs):
    """Estimate 2D geometric transformation parameters.

    You can determine the over-, well- and under-determined parameters
    with the least-squares method.

    Number of source must match number of destination coordinates.

    Parameters
    ----------
    ttype : str
        one of similarity, affine, projective, polynomial
    kwargs : array or int
        function parameters (src, dst, n, angle):

            NAME / TTYPE        FUNCTION PARAMETERS
            'similarity'        `src, `dst`
            'affine'            `src, `dst`
            'projective'        `src, `dst`
            'polynomial'        `src, `dst`, `order` (polynomial order)

        See examples section below for usage.

    Returns
    -------
    tform : :class:`GeometricTransformation`
        tform object containing the transformation parameters and providing
        access to forward and reverse transformation functions

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.transform import make_tform
    >>> src = np.array([0, 0, 10, 10]).reshape((2, 2))
    >>> dst = np.array([12, 14, 1, -20]).reshape((2, 2))
    >>> tform = estimate_transformation('similarity', src, dst)
    >>> print tform.params
    >>> print tform.reverse(tform.forward(src)) # == src
    >>> # warp image using the transformation
    >>> from skimage import data
    >>> image = data.camera()
    >>> warp(image, reverse_map=tform.forward)
    >>> warp(image, reverse_map=tform.reverse)

    """
    ttype = ttype.lower()
    if ttype not in ESTIMATED_TRANSFORMATIONS:
        raise NotImplemented('the transformation type \'%s\' is not'
                             'implemented' % ttype)
    matrix = ESTIMATED_TRANSFORMATIONS[ttype][0](*args, **kwargs)
    transform_func = ESTIMATED_TRANSFORMATIONS[ttype][1]
    return GeometricTransformation(ttype, matrix, transform_func)


def warp(image, reverse_map=None, map_args={}, output_shape=None, order=1,
         mode='constant', cval=0.):
    """Warp an image according to a given coordinate transformation.

    Parameters
    ----------
    image : 2-D array
        Input image.
    reverse_map : callable xy = f(xy, **kwargs)
        Reverse coordinate map.  A function that transforms a Px2 array of
        ``(x, y)`` coordinates in the *output image* into their corresponding
        coordinates in the *source image*.  Also see examples below.
    map_args : dict, optional
        Keyword arguments passed to `reverse_map`.
    output_shape : tuple (rows, cols)
        Shape of the output image generated.
    order : int
        Order of splines used in interpolation. See
        `scipy.ndimage.map_coordinates` for detail.
    mode : string
        How to handle values outside the image borders.  See
        `scipy.ndimage.map_coordinates` for detail.
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

    ## Construct transformed coordinates

    rows, cols = output_shape[:2]

    # Reshape grid coordinates into a (P, 2) array of (x, y) pairs
    tf_coords = np.indices((cols, rows), dtype=float).reshape(2, -1).T

    # Map each (x, y) pair to the source image according to
    # the user-provided mapping
    tf_coords = reverse_map(tf_coords, **map_args)

    # Reshape back to a (2, M, N) coordinate grid
    tf_coords = tf_coords.T.reshape((-1, cols, rows)).swapaxes(1, 2)

    # Place the y-coordinate mapping
    _stackcopy(coords[1, ...], tf_coords[0, ...])

    # Place the x-coordinate mapping
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
    import warnings
    warnings.warn('the homography function is deprecated; '
                  'use the `warp` and `tform` function instead',
                  category=DeprecationWarning)

    tform = GeometricTransformation('projective', H, geometric_transform)
    return warp(image, reverse_map=tform.reverse, output_shape=output_shape,
                order=order, mode=mode, cval=cval)
