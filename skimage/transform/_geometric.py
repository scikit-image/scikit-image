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


class GeometricTransform(object):
    """Perform geometric transformations on a set of coordinates.

    """
    def __call__(self, coords):
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
        raise NotImplementedError()

    def inverse(self, coords):
        """Apply inverse transformation.

        Parameters
        ----------
        coords : Nx2 array
            source coordinates

        Returns
        -------
        coords : Nx2 array
            transformed coordinates

        """
        raise NotImplementedError()

    def __add__(self, other):
        """Combine this transformation with another.

        """
        raise NotImplementedError()


class ProjectiveTransform(GeometricTransform):
    """Matrix transformation.

    Apply a projective transformation (homography) on coordinates.

    For each homogeneous coordinate :math:`\mathbf{x} = [x, y, 1]^T`, its
    target position is calculated by multiplying with the given matrix,
    :math:`H`, to give :math:`H \mathbf{x}`.  E.g., to rotate by theta degrees
    clockwise, the matrix should be

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
    matrix : 3x3 array, optional
        Homogeneous transformation matrix.

    """

    _coefs = range(8)

    def __init__(self, matrix=None):
        self._matrix = matrix

    @property
    def _inv_matrix(self):
        return np.linalg.inv(self._matrix)

    def _apply_mat(self, coords, matrix):
        coords = np.array(coords, copy=False, ndmin=2)

        x, y = np.transpose(coords)
        src = np.vstack((x, y, np.ones_like(x)))
        dst = np.dot(src.transpose(), matrix.transpose())

        # rescale to homogeneous coordinates
        dst[:, 0] /= dst[:, 2]
        dst[:, 1] /= dst[:, 2]

        return dst[:, :2]

    def __call__(self, coords):
        return self._apply_mat(coords, self._matrix)

    def inverse(self, coords):
        return self._apply_mat(coords, self._inv_matrix)

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

        # Select relevant columns, depending on coeffs
        A = A[:, self._coefs]

        b = np.hstack([xd, yd])

        H = np.zeros((3, 3))
        H.flat[self._coefs] = np.linalg.lstsq(A, b)[0]
        H[2, 2] = 1

        self._matrix = H

    def __add__(self, other):
        """Combine this transformation with another.

        """
        if isinstance(other, ProjectiveTransform):
            # combination of the same types result in a transformation of this
            # type again, otherwise use general projective transformation
            if type(self) == type(other):
                tform = self.__class__
            else:
                tform = ProjectiveTransform
            return tform(other._matrix.dot(self._matrix))
        else:
            raise TypeError("Cannot combine transformations of differing "
                            "types.")


class AffineTransform(ProjectiveTransform):

    """2D affine transformation of the form::

        X = a0*x + a1*y + a2 =
          = sx*x*cos(rotation) - sy*y*sin(rotation + shear) + a2

        Y = b0*x + b1*y + b2 =
          = sx*x*sin(rotation) + sy*y*cos(rotation + shear) + b2

    where ``sx`` and ``sy`` are zoom factors in the x and y directions,
    and the homogeneous transformation matrix is::

        [[a0  a1  a2]
         [b0  b1  b2]
         [0   0    1]]

    Parameters
    ----------
    matrix : 3x3 array, optional
        Homogeneous transformation matrix.

    """

    _coefs = range(6)

    def compose_implicit(self, scale=None, rotation=None, shear=None,
                    translation=None):
        """Set the transformation matrix with the implicit transformation
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
        if scale is None:
            scale = (1, 1)
        if rotation is None:
            rotation = 0
        if shear is None:
            shear = 0
        if translation is None:
            translation = (0, 0)

        sx, sy = scale
        self._matrix = np.array([
            [sx * math.cos(rotation), - sy * math.sin(rotation + shear), 0],
            [sx * math.sin(rotation),   sy * math.cos(rotation + shear), 0],
            [                      0,                                 0, 1]
        ])
        self._matrix[0:2, 2] = translation

    @property
    def scale(self):
        sx = math.sqrt(self._matrix[0, 0] ** 2 + self._matrix[1, 0] ** 2)
        sy = math.sqrt(self._matrix[0, 1] ** 2 + self._matrix[1, 1] ** 2)
        return sx, sy

    @property
    def rotation(self):
        return math.atan2(self._matrix[1, 0], self._matrix[0, 0])

    @property
    def shear(self):
        beta = math.atan2(- self._matrix[0, 1], self._matrix[1, 1])
        return beta - self.rotation

    @property
    def translation(self):
        return self._matrix[0:2, 2]


class SimilarityTransform(ProjectiveTransform):
    """2D similarity transformation of the form::

        X = a0*x + b0*y + a1 =
          = m*x*cos(rotation) + m*y*sin(rotation) + a1

        Y = b0*x + a0*y + b1 =
          = m*x*sin(rotation) + m*y*cos(rotation) + b1

    where ``m`` is a zoom factor and the homogeneous transformation matrix is::

        [[a0  b0  a1]
         [b0  a0  b1]
         [0   0    1]]

    Parameters
    ----------
    matrix : 3x3 array, optional
        Homogeneous transformation matrix.

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
        self._matrix = np.array([[a0, -b0, a1],
                                 [b0,  a0, b1],
                                 [ 0,   0,  1]])

    def compose_implicit(self, scale=None, rotation=None, translation=None):
        """Set the transformation matrix with the implicit transformation
        parameters.

        Parameters
        ----------
        scale : float, optional
            scale factor
        rotation : float, optional
            rotation angle in counter-clockwise direction
        translation : (tx, ty) as array, list or tuple, optional
            x, y translation parameters

        """
        if scale is None:
            scale = 1
        if rotation is None:
            rotation = 0
        if translation is None:
            translation = (0, 0)

        self._matrix = np.array([
            [math.cos(rotation), - math.sin(rotation), 0],
            [math.sin(rotation),   math.cos(rotation), 0],
            [                 0,                    0, 1]
        ])
        self._matrix *= scale
        self._matrix[0:2, 2] = translation

    @property
    def scale(self):
        if math.cos(self.rotation) == 0:
            # sin(self.rotation) == 1
            scale = self._matrix[0, 1]
        else:
            scale = self._matrix[0, 0] / math.cos(self.rotation)
        return scale

    @property
    def rotation(self):
        return math.atan2(self._matrix[1, 0], self._matrix[1, 1])

    @property
    def translation(self):
        return self._matrix[0:2, 2]


class PolynomialTransform(GeometricTransform):
    """2D transformation of the form::

        X = sum[j=0:n]( sum[i=0:j]( a_ji * x**(j - i) * y**i ))
        Y = sum[j=0:n]( sum[i=0:j]( b_ji * x**(j - i) * y**i ))

    TODO: Describe structure of coefficients.
          Shall we store it as a (2, M) ndarray?

    """

    def __init__(self, coeffs=None):
        """Create polynomial transformation.

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

    def __call__(self, coords):
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
        x = coords[:, 0]
        y = coords[:, 1]
        u = len(self.coeffs.ravel())
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

    def inverse(self, coords):
        raise Exception(
            'There is no explicit way to do the inverse polynomial '
            'transformation. Instead, estimate the inverse transformation '
            'parameters by exchanging source and destination coordinates,'
            'then apply the forward transformation.')


TRANSFORMATIONS = {
    'similarity': SimilarityTransform,
    'affine': AffineTransform,
    'projective': ProjectiveTransform,
    'polynomial': PolynomialTransform,
}


def estimate_transform(ttype, src, dst, **kwargs):
    """Estimate 2D geometric transformation parameters.

    You can determine the over-, well- and under-determined parameters
    with the least-squares method.

    Number of source must match number of destination coordinates.

    Parameters
    ----------
    ttype : {'similarity', 'affine', 'projective', 'polynomial'}
        Type of transform.
    kwargs : array or int
        Function parameters (src, dst, n, angle)::

            NAME / TTYPE        FUNCTION PARAMETERS
            'similarity'        `src, `dst`
            'affine'            `src, `dst`
            'projective'        `src, `dst`
            'polynomial'        `src, `dst`, `order` (polynomial order)

        Also see examples below.

    Returns
    -------
    tform : :class:`GeometricTransform`
        Transform object containing the transformation parameters and providing
        access to forward and inverse transformation functions.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage import transform as tf

    >>> # estimate transformation parameters
    >>> src = np.array([0, 0, 10, 10]).reshape((2, 2))
    >>> dst = np.array([12, 14, 1, -20]).reshape((2, 2))

    >>> tform = tf.estimate_transform('similarity', src, dst)

    >>> tform.inverse(tform(src)) # == src

    >>> # warp image using the estimated transformation
    >>> from skimage import data
    >>> image = data.camera()

    >>> warp(image, inverse_map=tform.inverse)

    >>> # create transformation with explicit parameters
    >>> tform2 = tf.SimilarityTransform()
    >>> tform2.compose_implicit(scale=1.1, rotation=1, translation=(10, 20))

    >>> # unite transformations, applied in order from left to right
    >>> tform3 = tform + tform2
    >>> tform3(src) # == tform2(tform(src))

    """
    ttype = ttype.lower()
    if ttype not in TRANSFORMATIONS:
        raise ValueError('the transformation type \'%s\' is not'
                         'implemented' % ttype)

    tform = TRANSFORMATIONS[ttype]()
    tform.estimate(src, dst, **kwargs)

    return tform


def matrix_transform(coords, matrix):
    """Apply 2D matrix transform.

    Parameters
    ----------
    coords : Nx2 array
        x, y coordinates to transform
    matrix : 3x3 array
        Homogeneous transformation matrix.

    Returns
    -------
    coords : Nx2 array
            transformed coordinates

    """
    return ProjectiveTransform(matrix)(coords)


def warp(image, inverse_map=None, map_args={}, output_shape=None, order=1,
         mode='constant', cval=0., reverse_map=None):
    """Warp an image according to a given coordinate transformation.

    Parameters
    ----------
    image : 2-D array
        Input image.
    inverse_map : transformation object, callable xy = f(xy, **kwargs)
        Inverse coordinate map. A function that transforms a Px2 array of
        ``(x, y)`` coordinates in the *output image* into their corresponding
        coordinates in the *source image*. In case of a transformation object
        its `inverse` method will be used as transformation function. Also see
        examples below.
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
    # Backward API compatibility
    if reverse_map is not None:
        inverse_map = reverse_map

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
    if callable(getattr(inverse_map, 'inverse', None)):
        inverse_map = inverse_map.inverse
    tf_coords = inverse_map(tf_coords, **map_args)

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
