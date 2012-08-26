import math
import numpy as np


class GeometricTransform(object):
    """Perform geometric transformations on a set of coordinates.

    """
    def __call__(self, coords):
        """Apply forward transformation.

        Parameters
        ----------
        coords : (N, 2) array
            Source coordinates.

        Returns
        -------
        coords : (N, 2) array
            Transformed coordinates.

        """
        raise NotImplementedError()

    def inverse(self, coords):
        """Apply inverse transformation.

        Parameters
        ----------
        coords : (N, 2) array
            Source coordinates.

        Returns
        -------
        coords : (N, 2) array
            Transformed coordinates.

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
        if matrix is None:
            # default to an identity transform
            matrix = np.eye(3)
        if matrix.shape != (3, 3):
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
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.

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
        Scale factors.
    rotation : float, optional
        Rotation angle in counter-clockwise direction as radians.
    shear : float, optional
        Shear angle in counter-clockwise direction as radians.
    translation : (tx, ty) as array, list or tuple, optional
        Translation parameters.

    """

    coeffs = range(6)

    def __init__(self, matrix=None, scale=None, rotation=None, shear=None,
                 translation=None):
        params = any(param is not None
                     for param in (scale, rotation, shear, translation))

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and "
                             "the implicit parameters at the same time.")
        elif matrix is not None:
            if matrix.shape != (3, 3):
                raise ValueError("Invalid shape of transformation matrix.")
            self._matrix = matrix
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
        else:
            # default to an identity transform
            self._matrix = np.eye(3)

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
        Scale factor.
    rotation : float, optional
        Rotation angle in counter-clockwise direction as radians.
    translation : (tx, ty) as array, list or tuple, optional
        x, y translation parameters.

    """

    def __init__(self, matrix=None, scale=None, rotation=None,
                 translation=None):
        params = any(param is not None
                     for param in (scale, rotation, translation))

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and "
                             "the implicit parameters at the same time.")
        elif matrix is not None:
            if matrix.shape != (3, 3):
                raise ValueError("Invalid shape of transformation matrix.")
            self._matrix = matrix
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
        else:
            # default to an identity transform
            self._matrix = np.eye(3)

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
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.

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
        if params is None:
            # default to transformation which preserves original coordinates
            params = np.array([[0, 1, 0], [0, 0, 1]])
        if params.shape[0] != 2:
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
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.
        order : int
            Polynomial order (number of coefficients is order + 1).

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
            Transformed coordinates.

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


TRANSFORMS = {
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
    if ttype not in TRANSFORMS:
        raise ValueError('the transformation type \'%s\' is not'
                         'implemented' % ttype)

    tform = TRANSFORMS[ttype]()
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
        Transformed coordinates.

    """
    return ProjectiveTransform(matrix)(coords)
