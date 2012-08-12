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
    Color images are stored as an ``(M, N, 3)`` or ``(M, N, 4)`` arrays.

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
        coords : (N, 2) array
            source coordinates

        Returns
        -------
        coords : (N, 2) array
            transformed coordinates

        """
        raise NotImplementedError()

    def inverse(self, coords):
        """Apply inverse transformation.

        Parameters
        ----------
        coords : (N, 2) array
            source coordinates

        Returns
        -------
        coords : (N, 2) array
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
    :math:`H`, to give :math:`H \mathbf{x}`::

      [[a0 a1 a2]
       [b0 b1 b2]
       [c0 c1 1 ]].

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
    matrix : (3, 3) array, optional
        Homogeneous transformation matrix.

    """

    coeffs = range(8)

    def __init__(self, matrix=None):
        if matrix is not None and matrix.shape != (3, 3):
            raise ValueError("invalid shape of transformation matrix")
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

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        The transformation is defined as::

            X = (a0*x + a1*y + a2) / (c0*x + c1*y + 1)
            Y = (b0*x + b1*y + b2) / (c0*x + c1*y + 1)

        These equations can be transformed to the following form::

            0 = a0*x + a1*y + a2 - c0*x*X - c1*y*X - X
            0 = b0*x + b1*y + b2 - c0*x*Y - c1*y*Y - Y

        which exist for each set of corresponding points, so we have a set of
        N * 2 equations. The coefficients appear linearly so we can write
        A x = 0, where::

            A   = [[x y 1 0 0 0 -x*X -y*X -X]
                   [0 0 0 x y 1 -x*Y -y*Y -Y]
                    ...
                    ...
                  ]
            x.T = [a0 a1 a2 b0 b1 b2 c0 c1 c3]

        In case of total least-squares the solution of this homogeneous system
        of equations is the right singular vector of A which corresponds to the
        smallest singular value normed by the coefficient c3.

        In case of the affine transformation the coefficients c0 and c1 are 0.
        Thus the system of equations is::

            A   = [[x y 1 0 0 0 -X]
                   [0 0 0 x y 1 -Y]
                    ...
                    ...
                  ]
            x.T = [a0 a1 a2 b0 b1 b2 c3]

        Parameters
        ----------
        src : (N, 2) array
            source coordinates
        dst : (N, 2) array
            destination coordinates

        """
        xs = src[:, 0]
        ys = src[:, 1]
        xd = dst[:, 0]
        yd = dst[:, 1]
        rows = src.shape[0]

        # params: a0, a1, a2, b0, b1, b2, c0, c1
        A = np.zeros((rows * 2, 9))
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
        A[:rows, 8] = xd
        A[rows:, 8] = yd

        # Select relevant columns, depending on params
        A = A[:, self.coeffs + [8]]

        _, _, V = np.linalg.svd(A)

        H = np.zeros((3, 3))
        # solution is right singular vector that corresponds to smallest
        # singular value
        H.flat[self.coeffs + [8]] = - V[-1, :-1] / V[-1, -1]
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
    matrix : (3, 3) array, optional
        Homogeneous transformation matrix.
    scale : (sx, sy) as array, list or tuple, optional
        scale factors
    rotation : float, optional
        rotation angle in counter-clockwise direction as radians
    shear : float, optional
        shear angle in counter-clockwise direction as radians
    translation : (tx, ty) as array, list or tuple, optional
        translation parameters

    """

    coeffs = range(6)

    def __init__(self, matrix=None, scale=None, rotation=None, shear=None,
                 translation=None):
        self._matrix = matrix
        params = any(param is not None
                     for param in (scale, rotation, shear, translation))

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and "
                             "the implicit parameters at the same time.")
        elif matrix is not None and matrix.shape != (3, 3):
            raise ValueError("Invalid shape of transformation matrix.")
        elif params:
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

        X = a0*x - b0*y + a1 =
          = m*x*cos(rotation) + m*y*sin(rotation) + a1

        Y = b0*x + a0*y + b1 =
          = m*x*sin(rotation) + m*y*cos(rotation) + b1

    where ``m`` is a zoom factor and the homogeneous transformation matrix is::

        [[a0  b0  a1]
         [b0  a0  b1]
         [0   0    1]]

    Parameters
    ----------
    matrix : (3, 3) array, optional
        Homogeneous transformation matrix.
    scale : float, optional
        scale factor
    rotation : float, optional
        rotation angle in counter-clockwise direction as radians
    translation : (tx, ty) as array, list or tuple, optional
        x, y translation parameters

    """

    def __init__(self, matrix=None, scale=None, rotation=None,
                 translation=None):
        self._matrix = matrix
        params = any(param is not None
                     for param in (scale, rotation, translation))

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and "
                             "the implicit parameters at the same time.")
        elif matrix is not None and matrix.shape != (3, 3):
            raise ValueError("Invalid shape of transformation matrix.")
        elif params:
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

    def estimate(self, src, dst):
        """Set the transformation matrix with the explicit parameters.

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        The transformation is defined as::

            X = a0*x - b0*y + a1
            Y = b0*x + a0*y + b1

        These equations can be transformed to the following form::

            0 = a0*x - b0*y + a1 - X
            0 = b0*x + a0*y + b1 - Y

        which exist for each set of corresponding points, so we have a set of
        N * 2 equations. The coefficients appear linearly so we can write
        A x = 0, where::

            A   = [[x 1 -y 0 -X]
                   [y 0  x 1 -Y]
                    ...
                    ...
                  ]
            x.T = [a0 a1 b0 b1 c3]

        In case of total least-squares the solution of this homogeneous system
        of equations is the right singular vector of A which corresponds to the
        smallest singular value normed by the coefficient c3.

        Parameters
        ----------
        src : (N, 2) array
            source coordinates
        dst : (N, 2) array
            destination coordinates

        """
        xs = src[:, 0]
        ys = src[:, 1]
        xd = dst[:, 0]
        yd = dst[:, 1]
        rows = src.shape[0]

        # params: a0, a1, b0, b1
        A = np.zeros((rows * 2, 5))
        A[:rows, 0] = xs
        A[:rows, 2] = - ys
        A[:rows, 1] = 1
        A[rows:, 2] = xs
        A[rows:, 0] = ys
        A[rows:, 3] = 1
        A[:rows, 4] = xd
        A[rows:, 4] = yd

        _, _, V = np.linalg.svd(A)

        # solution is right singular vector that corresponds to smallest
        # singular value
        a0, a1, b0, b1 = - V[-1, :-1] / V[-1, -1]

        self._matrix = np.array([[a0, -b0, a1],
                                 [b0,  a0, b1],
                                 [ 0,   0,  1]])

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

        X = sum[j=0:order]( sum[i=0:j]( a_ji * x**(j - i) * y**i ))
        Y = sum[j=0:order]( sum[i=0:j]( b_ji * x**(j - i) * y**i ))

    Parameters
    ----------
    params : (2, N) array, optional
        Polynomial coefficients where `N * 2 = (order + 1) * (order + 2)`. So,
        a_ji is defined in `params[0, :]` and b_ji in `params[1, :]`.

    """

    def __init__(self, params=None):
        if params is not None and params.shape[0] != 2:
            raise ValueError("invalid shape of transformation parameters")
        self._params = params

    def estimate(self, src, dst, order):
        """Set the transformation matrix with the explicit transformation
        parameters.

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        The transformation is defined as::

            X = sum[j=0:order]( sum[i=0:j]( a_ji * x**(j - i) * y**i ))
            Y = sum[j=0:order]( sum[i=0:j]( b_ji * x**(j - i) * y**i ))

        These equations can be transformed to the following form::

            0 = sum[j=0:order]( sum[i=0:j]( a_ji * x**(j - i) * y**i )) - X
            0 = sum[j=0:order]( sum[i=0:j]( b_ji * x**(j - i) * y**i )) - Y

        which exist for each set of corresponding points, so we have a set of
        N * 2 equations. The coefficients appear linearly so we can write
        A x = 0, where::

            A   = [[1 x y x**2 x*y y**2 ... 0 ...             0 -X]
                   [0 ...                 0 1 x y x**2 x*y y**2 -Y]
                    ...
                    ...
                  ]
            x.T = [a00 a10 a11 a20 a21 a22 ... ann
                   b00 b10 b11 b20 b21 b22 ... bnn c3]

        In case of total least-squares the solution of this homogeneous system
        of equations is the right singular vector of A which corresponds to the
        smallest singular value normed by the coefficient c3.

        Parameters
        ----------
        src : (N, 2) array
            source coordinates
        dst : (N, 2) array
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

        A = np.zeros((rows * 2, u + 1))
        pidx = 0
        for j in range(order + 1):
            for i in range(j + 1):
                A[:rows, pidx] = xs ** (j - i) * ys ** i
                A[rows:, pidx + u / 2] = xs ** (j - i) * ys ** i
                pidx += 1

        A[:rows, -1] = xd
        A[rows:, -1] = yd

        _, _, V = np.linalg.svd(A)

        # solution is right singular vector that corresponds to smallest
        # singular value
        params = - V[-1, :-1] / V[-1, -1]

        self._params = params.reshape((2, u / 2))

    def __call__(self, coords):
        """Apply forward transformation.

        Parameters
        ----------
        coords : (N, 2) array
            source coordinates

        Returns
        -------
        coords : (N, 2) array
            transformed coordinates

        """
        x = coords[:, 0]
        y = coords[:, 1]
        u = len(self._params.ravel())
        # number of coefficients -> u = (order + 1) * (order + 2)
        order = int((- 3 + math.sqrt(9 - 4 * (2 - u))) / 2)
        dst = np.zeros(coords.shape)

        pidx = 0
        for j in range(order + 1):
            for i in range(j + 1):
                dst[:, 0] += self._params[0, pidx] * x ** (j - i) * y ** i
                dst[:, 1] += self._params[1, pidx] * x ** (j - i) * y ** i
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
    with the total least-squares method.

    Number of source and destination coordinates must match.

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
    coords : (N, 2) array
        x, y coordinates to transform
    matrix : (3, 3) array
        Homogeneous transformation matrix.

    Returns
    -------
    coords : (N, 2) array
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
        Inverse coordinate map. A function that transforms a (N, 2) array of
        ``(x, y)`` coordinates in the *output image* into their corresponding
        coordinates in the *source image* (e.g. a transformation object or its
        inverse).
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
