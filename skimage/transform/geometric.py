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


class GeometricTransformation(object):

    def __init__(self, matrix=None):
        """Create geometric transformation which contains the transformation
        parameters and can perform forward and reverse transformations.

        Parameters
        ----------
        matrix : 3x3 array, optional
            homogeneous transformation matrix

        """
        self.matrix = matrix
        self.inverse_matrix = None

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
        if self.matrix is None:
            raise Exception('Transformation matrix must be set or estimated.')
        return geometric_transform(coords, self.matrix)

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
        if self.matrix is None:
            raise Exception('Transformation matrix must be set or estimated.')
        if self.inverse_matrix is None:
            self.inverse_matrix = np.linalg.inv(self.matrix)
        return geometric_transform(coords, self.inverse_matrix)

    def __add__(self, other):
        if type(self) == type(other):
            transformation = self.__class__
        else:
            transformation = GeometricTransformation
        return transformation(other.matrix.dot(self.matrix))


class SimilarityTransformation(GeometricTransformation):

    """2D similarity transformation of the following form:
        X = a0*x - b0*y + a1 =
          = m*x*cos(rotation) - m*y*sin(rotation) + a1
        Y = b0*x + a0*y + b1 =
          = m*x*sin(rotation) + m*y*cos(rotation) + b1
    where the homogeneous transformation matrix is:
        [[a0 -b0  a1]
         [b0  a0  b1]
         [0   0    1]]

    """

    def estimate(self, src, dst):
        """Set the transformation matrix with the estimated parameters of the
        given control points.

        Parameters
        ----------
        src : Nx2 array
            source coordinates
        dst : Nx2 array
            destination coordinates

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
        self.matrix = np.array([[a0, -b0, a1],
                                [b0,  a0, b1],
                                [ 0,   0,  1]])

    def from_params(self, scale, rotation, translation):
        """Set the transformation matrix with the explicit transformation
        parameters.

        Parameters
        ----------
        scale : float
            scale factor
        rotation : float
            rotation angle in counter-clockwise direction
        translation : (tx, ty) as array, list or tuple
            x, y translation parameters

        """
        self.matrix = np.array([
            [math.cos(rotation), - math.sin(rotation), 0],
            [math.sin(rotation),   math.cos(rotation), 0],
            [                 0,                    0, 1]
        ])
        self.matrix *= scale
        self.matrix[0:2, 2] = translation

    @property
    def scale(self):
        return self.matrix[0, 0] / math.cos(self.rotation)

    @property
    def rotation(self):
        return math.atan2(self.matrix[1, 0], self.matrix[1, 1])

    @property
    def translation(self):
        return self.matrix[0:2, 2]


class AffineTransformation(GeometricTransformation):

    """2D affine transformation of the following form
        X = a0*x + a1*y + a2 =
          = sx*x*cos(rotation) - sy*y*sin(rotation+shear) + a2
        Y = b0*x + b1*y + b2 =
          = sx*x*sin(rotation) + sy*y*cos(rotation+shear) + b2
    where the homogeneous transformation matrix is:
        [[a0  a1  a2]
         [b0  b1  b2]
         [0   0    1]]

    """

    def estimate(self, src, dst):
        """Set the transformation matrix with the estimated parameters of the
        given control points.

        Parameters
        ----------
        src : Nx2 array
            source coordinates
        dst : Nx2 array
            destination coordinates

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
        self.matrix = np.array([[a0, a1, a2],
                                [b0, b1, b2],
                                [0,   0,  1]])

    def from_params(self, scale, rotation, shear, translation):
        """Set the transformation matrix with the explicit transformation
        parameters.

        Parameters
        ----------
        scale : (sx, sy) as array, list or tuple
            scale factors
        rotation : float
            rotation angle in counter-clockwise direction
        shear : float
            shear angle in counter-clockwise direction
        translation : (tx, ty) as array, list or tuple
            translation parameters

        """
        sx, sy = scale
        self.matrix = np.array([
            [sx * math.cos(rotation), - sy * math.sin(rotation + shear), 0],
            [sx * math.sin(rotation),   sy * math.cos(rotation + shear), 0],
            [                 0,                                      0, 1]
        ])
        self.matrix[0:2, 2] = translation

    @property
    def scale(self):
        sx = math.sqrt(self.matrix[0, 0] ** 2 + self.matrix[1, 0] ** 2)
        sy = math.sqrt(self.matrix[0, 1] ** 2 + self.matrix[1, 1] ** 2)
        return sx, sy

    @property
    def rotation(self):
        return math.atan2(self.matrix[1, 0], self.matrix[0, 0])

    @property
    def shear(self):
        beta = math.atan2(- self.matrix[0, 1], self.matrix[1, 1])
        return beta - self.rotation

    @property
    def translation(self):
        return self.matrix[0:2, 2]


class ProjectiveTransformation(GeometricTransformation):

    """2D projective transformation of the following form
        X = (a0 + a1*x + a2*y) / (c0*x + c1*y + 1)
        Y = (b0 + b1*x + b2*y) / (c0*x + c1*y + 1)
    where the homogeneous transformation matrix is:
        [[a0  a1  a2]
         [b0  b1  b2]
         [c0  c1   1]]

    """

    def estimate(self, src, dst):
        """Set the transformation matrix with the explicit transformation
        parameters.

        Parameters
        ----------
        src : Nx2 array
            source coordinates
        dst : Nx2 array
            destination coordinates

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
        self.matrix = np.array([[a0, a1, a2],
                                [b0, b1, b2],
                                [c0, c1,  1]])


class PolynomialTransformation(GeometricTransformation):

    """2D affine transformation of the following form
        X = sum[j=0:n]( sum[i=0:j]( a_ji * x**(j - i) * y**i ))
        Y = sum[j=0:n]( sum[i=0:j]( b_ji * x**(j - i) * y**i ))

    """

    def __init__(self, coeffs=None):
        """Create polynomial transformation which contains the transformation
        parameters and can perform forward and reverse transformations.

        Parameters
        ----------
        coeffs : array, optional
            polynomial coefficients

        """
        self.coeffs = coeffs

    def estimate(self, src, dst, order):
        """Set the transformation matrix with the explicit transformation
        parameters.

        Parameters
        ----------
        src : Nx2 array
            source coordinates
        dst : Nx2 array
            destination coordinates
        order : int
            polynomial order (number of coefficients is order + 1)

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

        self.coeffs = np.linalg.lstsq(A, b)[0]

    def forward(self, coords):
        x = coords[:, 0]
        y = coords[:, 1]
        u = len(self.coeffs)
        # number of coefficients -> u = (order + 1) * (order + 2)
        order = int((- 3 + math.sqrt(9 - 4 * (2 - u))) / 2)
        dst = np.zeros(coords.shape)

        pidx = 0
        for j in xrange(order + 1):
            for i in xrange(j + 1):
                dst[:, 0] += self.coeffs[pidx] * x ** (j - i) * y ** i
                dst[:, 1] += self.coeffs[pidx + u / 2] * x ** (j - i) * y ** i
                pidx += 1

        return dst

    def reverse(self, coords):
        raise Exception(
            'There is no explicit way to do the reverse polynomial '
            'transformation. Instead determine the reverse transformation '
            'parameters by exchanging source and destination coordinates.'
            'Then apply the forward transformation.')


TRANSFORMATIONS = {
    'similarity': SimilarityTransformation,
    'affine': AffineTransformation,
    'projective': ProjectiveTransformation,
    'polynomial': PolynomialTransformation,
}


def estimate_transformation(ttype, src, dst, order=None):
    """Estimate 2D geometric transformation parameters.

    You can determine the over-, well- and under-determined parameters
    with the least-squares method.

    Number of source must match number of destination coordinates.

    Parameters
    ----------
    ttype : str
        one of similarity, affine, projective, polynomial
    kwargs :: array or int
        function parameters (src, dst, n, angle):

            NAME / TTYPE        FUNCTION PARAMETERS
            'similarity'        `src, `dst`
            'affine'            `src, `dst`
            'projective'        `src, `dst`
            'polynomial'        `src, `dst`, `order` (polynomial order)

        See examples section below for usage.

    Returns
    -------
    tform : subclass of :class:`GeometricTransformation`
        tform object containing the transformation parameters and providing
        access to forward and reverse transformation functions

    Examples
    --------
    >>> import numpy as np
    >>> from skimage import transform as tf
    >>> # estimate transformation parameters
    >>> src = np.array([0, 0, 10, 10]).reshape((2, 2))
    >>> dst = np.array([12, 14, 1, -20]).reshape((2, 2))
    >>> tform = tf.estimate_transformation('similarity', src, dst)
    >>> tform.matrix
    >>> tform.reverse(tform.forward(src)) # == src
    >>> # warp image using the estimated transformation
    >>> from skimage import data
    >>> image = data.camera()
    >>> tf.warp(image, tform) # == warp(image, reverse_map=tform.reverse)
    >>> tf.warp(image, reverse_map=tform.forward)
    >>> # create transformation with explicit parameters
    >>> tform2 = tf.SimilarityTransformation()
    >>> scale = 1.1
    >>> rotation = 1
    >>> translation = (10, 20)
    >>> tform2.from_params(scale, rotation, translation)
    >>> # unite transformations, applied in order from left to right
    >>> tform3 = tform + tform2
    >>> tform3.forward(src) # == tform2.forward(tform.forward(src))

    """
    ttype = ttype.lower()
    if ttype not in TRANSFORMATIONS:
        raise ValueError('the transformation type \'%s\' is not'
                         'implemented' % ttype)
    args = [src, dst]
    if order is not None and ttype == 'polynomial':
        args.append(order)
    tform = TRANSFORMATIONS[ttype]()
    tform.estimate(*args)
    return tform


def warp(image, reverse_map=None, map_args={}, output_shape=None, order=1,
         mode='constant', cval=0.):
    """Warp an image according to a given coordinate transformation.

    Parameters
    ----------
    image : 2-D array
        Input image.
    reverse_map : transformation object, callable xy = f(xy, **kwargs)
        Reverse coordinate map. A function that transforms a Px2 array of
        ``(x, y)`` coordinates in the *output image* into their corresponding
        coordinates in the *source image*. In case of a transformation object
        its `reverse` method will be used as transformation function. Also see
        examples below.
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
    if callable(getattr(reverse_map, 'reverse', None)):
        reverse_map = reverse_map.reverse
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

    tform = ProjectiveTransformation(H)
    return warp(image, reverse_map=tform.reverse, output_shape=output_shape,
                order=order, mode=mode, cval=cval)
