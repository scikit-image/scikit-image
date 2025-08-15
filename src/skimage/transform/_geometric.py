from copy import copy
import math
import textwrap
from abc import ABC, abstractmethod
from typing import Self
import warnings

import numpy as np
from scipy import spatial

from .._shared.utils import (
    safe_as_int,
    _deprecate_estimate,
    _update_from_estimate_docstring,
    _deprecate_inherited_estimate,
    FailedEstimation,
)
from .._shared.compat import NP_COPY_IF_NEEDED


def _affine_matrix_from_vector(v):
    """Affine matrix from linearized (d, d + 1) matrix entries."""
    nparam = v.size
    # solve for d in: d * (d + 1) = nparam
    d = (1 + np.sqrt(1 + 4 * nparam)) / 2 - 1
    dimensionality = int(np.round(d))  # round to prevent approx errors
    if d != dimensionality:
        raise ValueError(
            'Invalid number of elements for ' f'linearized matrix: {nparam}'
        )
    matrix = np.eye(dimensionality + 1)
    matrix[:-1, :] = np.reshape(v, (dimensionality, dimensionality + 1))
    return matrix


def _calc_center_normalize(points, scaling='rms'):
    """Calculate transformation `matrix` to center and normalize image points.

    Points are an array of shape (N, D).

    For `scaling` of 'raw', transformation returned `matrix` will be ``np.eye(D
    + 1)``.  For other values of `scaling`, `matrix` expresses a two-step
    translation and scaling procedure.  Points transformed with this `matrix`
    usually give better conditioning for fundamental matrix estimation than the
    original `points` [1]_.

    The two steps of transformation, for `scaling` other than 'raw', are:

    * Center the image points, such that the new coordinate system has its
      origin at the centroid of the image points.
    * Normalize the image points, such that the mean coordinate value of the
      centered points is 1 (`scaling` == 'rms') or such that the
      mean distance from the points to the origin of the coordinate system is
      ``sqrt(D)`` (`scaling` == 'mrs').

    If `scaling` != 'raw' and the points are all identical, the returned
    `matrix` will be all ``np.nan``.

    The 'mrs' scaling corresponds to the isotropic transformation
    algorithm in [1]_. 'rms' is the default, and gives very similar
    conditioning.

    Parameters
    ----------
    points : (N, D) array
        The coordinates of the image points.
    scaling : {'rms', 'mrs', 'raw'}, optional
        Scaling algorithm adjusting for magnitude of `points` after applying
        calculated translation. See above for explanation.

    Returns
    -------
    matrix : (D+1, D+1) array_like
        The transformation matrix to obtain the new points.

    References
    ----------
    .. [1] Hartley, Richard I. "In defense of the eight-point algorithm."
           Pattern Analysis and Machine Intelligence, IEEE Transactions on 19.6
           (1997): 580-593.

    """
    n, d = points.shape
    scaling = scaling.lower()
    matrix = np.eye(d + 1)
    if scaling == 'raw':
        return matrix
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    if scaling == 'rms':
        divisor = np.sqrt(np.mean(centered**2))
    elif scaling == 'mrs':
        divisor = np.mean(np.sqrt(np.sum(centered**2, axis=1))) / np.sqrt(d)
    else:
        raise ValueError(f'Unexpected "scaling" of "{scaling}"')

    # if all the points are the same, the transformation matrix cannot be
    # created. We return an equivalent matrix with np.nans as sentinel values.
    # This obviates the need for try/except blocks in functions calling this
    # one, and those are only needed when actual 0 is reached, rather than some
    # small value; ie, we don't need to worry about numerical stability here,
    # only actual 0.
    if divisor == 0:
        return matrix + np.nan

    matrix[:d, d] = -centroid
    matrix[:d, :] /= divisor
    return matrix


def _center_and_normalize_points(points, scaling='rms'):
    """Convenience function to calculate and apply scaling

    See: :func:`_calc_center_normalize` for details of the algorithm.
    """
    matrix = _calc_center_normalize(points, scaling)
    if not np.all(np.isfinite(matrix)):
        return matrix + np.nan, np.full_like(points, np.nan)
    return matrix, _apply_homogeneous(matrix, points)


def _apply_homogeneous(matrix, points):
    """Transform (N, D) `points` array with homogeneous (D+1, D+1) `matrix`.

    Parameters
    ----------
    matrix : (D+1, D+1) array_like
        The transformation matrix to obtain the new points. Note that any
        object with an `__array__` method [1]_ that returns a matrix with the
        correct dimensions can be used as input here. This includes all
        subclasses of :class:`ProjectiveTransform`, for example.
    points : (N, D) array
        The coordinates of the image points.

    Returns
    -------
    new_points : (N, D) array
        The transformed image points.

    References
    ----------
    .. [1]:
        https://numpy.org/doc/stable/user/basics.interoperability.html#using-arbitrary-objects-in-numpy
    """
    points = np.array(points, copy=NP_COPY_IF_NEEDED, ndmin=2)
    points_h = _append_homogeneous_dim(points)
    new_points_h = points_h @ matrix.T
    # We divide by the last dimension of the homogeneous
    # coordinate matrix. In order to avoid division by zero,
    # we replace exact zeros in this column with a very small number.
    divs = new_points_h[:, -1]
    divs = np.where(divs == 0, np.finfo(float).eps, divs)
    return new_points_h[:, :-1] / divs[:, None]


def _append_homogeneous_dim(points):
    """Append a column of ones to the right of `points`.

    This creates the representation of the points in the homogeneous coordinate
    space used by homogeneous matrix transforms.

    Parameters
    ----------
    points : array, shape (N, D)
        The input coordinates, where N is the number of points and D is the
        dimension of the coordinate space.

    Returns
    -------
    points_h : array, shape (N, D+1)
        The same points as homogeneous coordinates.
    """
    return np.hstack((points, np.ones((len(points), 1))))


def _umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.

    Parameters
    ----------
    src : (M, N) array_like
        Source coordinates.
    dst : (M, N) array_like
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.

    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.

    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`

    """
    src = np.asarray(src)
    dst = np.asarray(dst)

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    # Matrix rank calculation from SVD (see numpy.linalg._linalg::matrix_rank code).
    # (this does SVD to check for small singular values, replicated here).
    tol = S.max() * np.max(A.shape) * np.finfo(float).eps
    rank = np.count_nonzero(S > tol)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale

    return T


class _GeometricTransform(ABC):
    """Abstract base class for geometric transformations."""

    @abstractmethod
    def __call__(self, coords):
        """Apply forward transformation.

        Parameters
        ----------
        coords : (N, 2) array_like
            Source coordinates.

        Returns
        -------
        coords : (N, 2) array
            Destination coordinates.

        """

    @property
    @abstractmethod
    def inverse(self):
        """Return a transform object representing the inverse."""

    def residuals(self, src, dst):
        """Determine residuals of transformed destination coordinates.

        For each transformed source coordinate the Euclidean distance to the
        respective destination coordinate is determined.

        Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.

        Returns
        -------
        residuals : (N,) array
            Residual for coordinate.

        """
        return np.sqrt(np.sum((self(src) - dst) ** 2, axis=1))

    @classmethod
    @abstractmethod
    def identity(cls, dimensionality=None):
        """Identity transform

        Parameters
        ----------
        dimensionality : {None, 2}, optional
            This transform only allows dimensionality of 2, where None
            corresponds to 2. The parameter exists for compatibility with other
            transforms.

        Returns
        -------
        tform : transform
            Transform such that ``np.all(tform(pts) == pts)``.
        """

    @classmethod
    def _prepare_estimation(cls, src, dst):
        """Create identity transform and make sure points are arrays."""
        src = np.asarray(src)
        dst = np.asarray(dst)
        return cls.identity(src.shape[1]), src, dst

    @classmethod
    def from_estimate(cls, src, dst, *args, **kwargs) -> Self | FailedEstimation:
        r"""Estimate transform.

        Parameters
        ----------
        src : (N, M) array_like
            Source coordinates.
        dst : (N, M) array_like
            Destination coordinates.
        \*args : sequence
            Any other positional arguments.
        \*\*kwargs : dict
            Any other keyword arguments.

        Returns
        -------
        tf : Self or ``FailedEstimation``
            An instance of the transformation if the estimation succeeded.
            Otherwise, we return a special ``FailedEstimation`` object to
            signal a failed estimation. Testing the truth value of the failed
            estimation object will return ``False``. E.g.

            .. code-block:: python

                tf = TransformClass.from_estimate(...)
                if not tf:
                    raise RuntimeError(f"Failed estimation: {tf}")
        """
        return _from_estimate(cls, src, dst, *args, **kwargs)


def _from_estimate(cls, src, dst, *args, **kwargs):
    """Detached function for from_estimate base implementation."""
    tf, src, dst = cls._prepare_estimation(src, dst)
    msg = tf._estimate(src, dst, *args, **kwargs)
    return tf if msg is None else FailedEstimation(f'{cls.__name__}: {msg}')


class _HMatrixTransform(_GeometricTransform):
    """Transform accepting homogeneous matrix as input."""

    def __init__(self, matrix=None, *, dimensionality=None):
        if matrix is None:
            d = 2 if dimensionality is None else dimensionality
            matrix = np.eye(d + 1)
        else:
            matrix = np.asarray(matrix)
        self._check_matrix(matrix, dimensionality)
        self._check_dims(matrix.shape[0] - 1)
        self.params = matrix

    def _check_matrix(self, matrix, dimensionality):
        if dimensionality is not None:
            if dimensionality != matrix.shape[0] - 1:
                raise ValueError(
                    f'Dimensionality {dimensionality} does not match matrix '
                    f'{matrix}'
                )
        m = matrix.shape[0]
        if matrix.shape != (m, m):
            raise ValueError("Invalid shape of transformation matrix")

    def _check_dims(self, d):
        if d == 2:
            return
        raise NotImplementedError(
            f'Input for {type(self)} should result in 2D transform'
        )

    @classmethod
    def identity(cls, dimensionality=None):
        """Identity transform

        Parameters
        ----------
        dimensionality : {None, 2}, optional
            This transform only allows dimensionality of 2, where None
            corresponds to 2. The parameter exists for compatibility with other
            transforms.

        Returns
        -------
        tform : transform
            Transform such that ``np.all(tform(pts) == pts)``.
        """
        d = 2 if dimensionality is None else dimensionality
        return cls(matrix=np.eye(d + 1))

    @property
    def dimensionality(self):
        return self.matrix.shape[0] - 1


class FundamentalMatrixTransform(_HMatrixTransform):
    """Fundamental matrix transformation.

    The fundamental matrix relates corresponding points between a pair of
    uncalibrated images. The matrix transforms homogeneous image points in one
    image to epipolar lines in the other image.

    The fundamental matrix is only defined for a pair of moving images. In the
    case of pure rotation or planar scenes, the homography describes the
    geometric relation between two images (`ProjectiveTransform`). If the
    intrinsic calibration of the images is known, the essential matrix describes
    the metric relation between the two images (`EssentialMatrixTransform`).

    Notes
    -----
    See [1]_ and [2]_ for details of the estimation procedure.  [2]_ is a good
    place to start.

    References
    ----------
    .. [1] Hartley, Richard, and Andrew Zisserman. Multiple view geometry in
           computer vision. Cambridge university press, 2003.
    .. [2] Zhang, Zhengyou. "Determining the epipolar geometry and its
           uncertainty: A review." International journal of computer vision 27
           (1998): 161-195.
           :DOI:`10.1023/A:1007941100561`
           https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/RR-2927.pdf

    Parameters
    ----------
    matrix : (3, 3) array_like, optional
        Fundamental matrix.
    dimensionality : int, optional
        Fallback number of dimensions when `matrix` not specified, in which
        case, must equal 2 (the default).

    Attributes
    ----------
    params : (3, 3) array
        Fundamental matrix.

    Examples
    --------
    >>> import numpy as np
    >>> import skimage as ski

    Define source and destination points:

    >>> src = np.array([1.839035, 1.924743,
    ...                 0.543582, 0.375221,
    ...                 0.473240, 0.142522,
    ...                 0.964910, 0.598376,
    ...                 0.102388, 0.140092,
    ...                15.994343, 9.622164,
    ...                 0.285901, 0.430055,
    ...                 0.091150, 0.254594]).reshape(-1, 2)
    >>> dst = np.array([1.002114, 1.129644,
    ...                 1.521742, 1.846002,
    ...                 1.084332, 0.275134,
    ...                 0.293328, 0.588992,
    ...                 0.839509, 0.087290,
    ...                 1.779735, 1.116857,
    ...                 0.878616, 0.602447,
    ...                 0.642616, 1.028681]).reshape(-1, 2)

    Estimate the transformation matrix:

    >>> tform = ski.transform.FundamentalMatrixTransform.from_estimate(
    ...      src, dst)
    >>> tform.params
    array([[-0.21785884,  0.41928191, -0.03430748],
           [-0.07179414,  0.04516432,  0.02160726],
           [ 0.24806211, -0.42947814,  0.02210191]])

    Compute the Sampson distance:

    >>> tform.residuals(src, dst)
    array([0.0053886 , 0.00526101, 0.08689701, 0.01850534, 0.09418259,
           0.00185967, 0.06160489, 0.02655136])

    Apply inverse transformation:

    >>> tform.inverse(dst)
    array([[-0.0513591 ,  0.04170974,  0.01213043],
           [-0.21599496,  0.29193419,  0.00978184],
           [-0.0079222 ,  0.03758889, -0.00915389],
           [ 0.14187184, -0.27988959,  0.02476507],
           [ 0.05890075, -0.07354481, -0.00481342],
           [-0.21985267,  0.36717464, -0.01482408],
           [ 0.01339569, -0.03388123,  0.00497605],
           [ 0.03420927, -0.1135812 ,  0.02228236]])

    The estimation can fail - for example, if all the input or output points
    are the same.  If this happens, you will get a transform that is not
    "truthy" - meaning that ``bool(tform)`` is ``False``:

    >>> # A successfully estimated model is truthy (applying ``bool()``
    >>> # gives ``True``):
    >>> if tform:
    ...     print("Estimation succeeded.")
    Estimation succeeded.
    >>> # Not so for a degenerate transform with identical points.
    >>> bad_src = np.ones((8, 2))
    >>> bad_tform = ski.transform.FundamentalMatrixTransform.from_estimate(
    ...      bad_src, dst)
    >>> if not bad_tform:
    ...     print("Estimation failed.")
    Estimation failed.

    Trying to use this failed estimation transform result will give a suitable
    error:

    >>> bad_tform.params  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      ...
    FailedEstimationAccessError: No attribute "params" for failed estimation ...

    """

    scaling = 'rms'

    def __call__(self, coords):
        """Apply forward transformation.

        Parameters
        ----------
        coords : (N, 2) array_like
            Source coordinates.

        Returns
        -------
        coords : (N, 3) array
            Epipolar lines in the destination image.

        """
        return _append_homogeneous_dim(coords) @ self.params.T

    @property
    def inverse(self):
        """Return a transform object representing the inverse.

        See Hartley & Zisserman, Ch. 8: Epipolar Geometry and the Fundamental
        Matrix, for an explanation of why F.T gives the inverse.

        """
        return type(self)(matrix=self.params.T)

    def _setup_constraint_matrix(self, src, dst):
        """Setup and solve the homogeneous epipolar constraint matrix::

            dst' * F * src = 0.

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.

        Returns
        -------
        F_normalized : (3, 3) array
            The normalized solution to the homogeneous system. If the system
            is not well-conditioned, this matrix contains NaNs.
        src_matrix : (3, 3) array
            The transformation matrix to obtain the normalized source
            coordinates.
        dst_matrix : (3, 3) array
            The transformation matrix to obtain the normalized destination
            coordinates.

        """
        src = np.asarray(src)
        dst = np.asarray(dst)
        if src.shape != dst.shape:
            raise ValueError('src and dst shapes must be identical.')
        if src.shape[0] < 8:
            raise ValueError('src.shape[0] must be equal or larger than 8.')

        # Center and normalize image points for better numerical stability.
        src_matrix = _calc_center_normalize(src, self.scaling)
        dst_matrix = _calc_center_normalize(dst, self.scaling)
        if np.any(np.isnan(src_matrix + dst_matrix)):
            self.params = np.full((3, 3), np.nan)
            return 3 * [np.full((3, 3), np.nan)]
        src_h = _append_homogeneous_dim(_apply_homogeneous(src_matrix, src))
        dst_h = _append_homogeneous_dim(_apply_homogeneous(dst_matrix, dst))

        # Setup homogeneous linear equation as dst' * F * src = 0.
        # Hartley notation u -> src[:, 0], v -> src[:, 1],
        # u' -> dst[:, 0], v' -> dst[:, 1].  Required output cols are:
        # uu', vu', u', uv', vv', v', u, v, 1
        cols = [(d_v * s_v) for d_v in dst_h.T for s_v in src_h.T]
        A = np.stack(cols, axis=1)

        # Solve for the nullspace of the constraint matrix.
        _, _, V = np.linalg.svd(A)
        F_normalized = V[-1, :].reshape(3, 3)

        return F_normalized, src_matrix, dst_matrix

    @classmethod
    def from_estimate(cls, src, dst):
        """Estimate fundamental matrix using 8-point algorithm.

        The 8-point algorithm requires at least 8 corresponding point pairs.

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.

        Returns
        -------
        tf : Self or ``FailedEstimation``
            An instance of the transformation if the estimation succeeded.
            Otherwise, we return a special ``FailedEstimation`` object to
            signal a failed estimation. Testing the truth value of the failed
            estimation object will return ``False``. E.g.

            .. code-block:: python

                tf = FundamentalMatrixTransform.from_estimate(...)
                if not tf:
                    raise RuntimeError(f"Failed estimation: {tf}")

        Raises
        ------
        ValueError
            If `src` has fewer than 8 rows.
        """
        return super().from_estimate(src, dst)

    def _estimate(self, src, dst):
        F_normalized, src_matrix, dst_matrix = self._setup_constraint_matrix(src, dst)
        if np.any(np.isnan(F_normalized + src_matrix + dst_matrix)):
            return 'Scaling failed for input points'

        # Enforcing the internal constraint that two singular values must be
        # non-zero and one must be zero (rank 2).
        U, S, V = np.linalg.svd(F_normalized)
        S[2] = 0
        F = U @ np.diag(S) @ V

        self.params = dst_matrix.T @ F @ src_matrix

        return None

    def residuals(self, src, dst):
        """Compute the Sampson distance.

        The Sampson distance is the first approximation to the geometric error.

        Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.

        Returns
        -------
        residuals : (N,) array
            Sampson distance.

        """
        src_homogeneous = _append_homogeneous_dim(src)
        dst_homogeneous = _append_homogeneous_dim(dst)

        F_src = self.params @ src_homogeneous.T
        Ft_dst = self.params.T @ dst_homogeneous.T

        dst_F_src = np.sum(dst_homogeneous * F_src.T, axis=1)

        return np.abs(dst_F_src) / np.sqrt(
            F_src[0] ** 2 + F_src[1] ** 2 + Ft_dst[0] ** 2 + Ft_dst[1] ** 2
        )

    @_deprecate_estimate
    def estimate(self, src, dst):
        """Estimate fundamental matrix using 8-point algorithm.

        The 8-point algorithm requires at least 8 corresponding point pairs for
        a well-conditioned solution, otherwise the over-determined solution is
        estimated.

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """
        return self._estimate(src, dst) is None


class EssentialMatrixTransform(FundamentalMatrixTransform):
    """Essential matrix transformation.

    The essential matrix relates corresponding points between a pair of
    calibrated images. The matrix transforms normalized, homogeneous image
    points in one image to epipolar lines in the other image.

    The essential matrix is only defined for a pair of moving images capturing a
    non-planar scene. In the case of pure rotation or planar scenes, the
    homography describes the geometric relation between two images
    (`ProjectiveTransform`). If the intrinsic calibration of the images is
    unknown, the fundamental matrix describes the projective relation between
    the two images (`FundamentalMatrixTransform`).

    References
    ----------
    .. [1] Hartley, Richard, and Andrew Zisserman. Multiple view geometry in
           computer vision. Cambridge university press, 2003.

    Parameters
    ----------
    rotation : (3, 3) array_like, optional
        Rotation matrix of the relative camera motion.
    translation : (3, 1) array_like, optional
        Translation vector of the relative camera motion. The vector must
        have unit length.
    matrix : (3, 3) array_like, optional
        Essential matrix.
    dimensionality : int, optional
        Fallback number of dimensions when `matrix` not specified, in which
        case, must equal 2 (the default).

    Attributes
    ----------
    params : (3, 3) array
        Essential matrix.

    Examples
    --------
    >>> import numpy as np
    >>> import skimage as ski
    >>>
    >>> tform = ski.transform.EssentialMatrixTransform(
    ...     rotation=np.eye(3), translation=np.array([0, 0, 1])
    ... )
    >>> tform.params
    array([[ 0., -1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  0.]])
    >>> src = np.array([[ 1.839035, 1.924743],
    ...                 [ 0.543582, 0.375221],
    ...                 [ 0.47324 , 0.142522],
    ...                 [ 0.96491 , 0.598376],
    ...                 [ 0.102388, 0.140092],
    ...                 [15.994343, 9.622164],
    ...                 [ 0.285901, 0.430055],
    ...                 [ 0.09115 , 0.254594]])
    >>> dst = np.array([[1.002114, 1.129644],
    ...                 [1.521742, 1.846002],
    ...                 [1.084332, 0.275134],
    ...                 [0.293328, 0.588992],
    ...                 [0.839509, 0.08729 ],
    ...                 [1.779735, 1.116857],
    ...                 [0.878616, 0.602447],
    ...                 [0.642616, 1.028681]])
    >>> tform = ski.transform.EssentialMatrixTransform.from_estimate(src, dst)
    >>> tform.residuals(src, dst)
    array([0.42455187, 0.01460448, 0.13847034, 0.12140951, 0.27759346,
           0.32453118, 0.00210776, 0.26512283])

    The estimation can fail - for example, if all the input or output points
    are the same.  If this happens, you will get a transform that is not
    "truthy" - meaning that ``bool(tform)`` is ``False``:

    >>> # A successfully estimated model is truthy (applying ``bool()``
    >>> # gives ``True``):
    >>> if tform:
    ...     print("Estimation succeeded.")
    Estimation succeeded.
    >>> # Not so for a degenerate transform with identical points.
    >>> bad_src = np.ones((8, 2))
    >>> bad_tform = ski.transform.EssentialMatrixTransform.from_estimate(
    ...      bad_src, dst)
    >>> if not bad_tform:
    ...     print("Estimation failed.")
    Estimation failed.

    Trying to use this failed estimation transform result will give a suitable
    error:

    >>> bad_tform.params  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      ...
    FailedEstimationAccessError: No attribute "params" for failed estimation ...
    """

    # Threshold for determinant of rotation matrix.
    _rot_det_tol = 1e-6

    # Threshold for difference of translation vector from unit length.
    _trans_len_tol = 1e-6

    def __init__(
        self, *, rotation=None, translation=None, matrix=None, dimensionality=None
    ):
        n_rt_none = sum(p is None for p in (rotation, translation))
        if n_rt_none == 1:
            raise ValueError(
                "Both rotation and translation required when one is specified."
            )
        elif n_rt_none == 0:
            if matrix is not None:
                raise ValueError(
                    "Do not specify rotation or translation when "
                    "matrix is specified."
                )
            matrix = self._rt2matrix(rotation, translation)
        super().__init__(matrix=matrix, dimensionality=dimensionality)

    def _rt2matrix(self, rotation, translation):
        rotation = np.asarray(rotation)
        translation = np.asarray(translation)
        if rotation.shape != (3, 3):
            raise ValueError("Invalid shape of rotation matrix")
        if abs(np.linalg.det(rotation) - 1) > self._rot_det_tol:
            raise ValueError("Rotation matrix must have unit determinant")
        if translation.size != 3:
            raise ValueError("Invalid shape of translation vector")
        if abs(np.linalg.norm(translation) - 1) > self._trans_len_tol:
            raise ValueError("Translation vector must have unit length")
        # Matrix representation of the cross product for t.
        t0, t1, t2 = translation
        t_arr = np.array([[0, -t2, t1], [t2, 0, -t0], [-t1, t0, 0]], dtype=float)
        return t_arr @ rotation

    @classmethod
    def from_estimate(cls, src, dst):
        """Estimate essential matrix using 8-point algorithm.

        The 8-point algorithm requires at least 8 corresponding point pairs for
        a well-conditioned solution, otherwise the over-determined solution is
        estimated.

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.

        Returns
        -------
        tf : Self or ``FailedEstimation``
            An instance of the transformation if the estimation succeeded.
            Otherwise, we return a special ``FailedEstimation`` object to
            signal a failed estimation. Testing the truth value of the failed
            estimation object will return ``False``. E.g.

            .. code-block:: python

                tf = EssentialMatrixTransform.from_estimate(...)
                if not tf:
                    raise RuntimeError(f"Failed estimation: {tf}")

        Raises
        ------
        ValueError
            If `src` has fewer than 8 rows.

        """
        return super().from_estimate(src, dst)

    def _estimate(self, src, dst):
        E_normalized, src_matrix, dst_matrix = self._setup_constraint_matrix(src, dst)
        if np.any(np.isnan(E_normalized + src_matrix + dst_matrix)):
            return 'Scaling failed for input points'

        # Enforcing the internal constraint that two singular values must be
        # equal and one must be zero.
        U, S, V = np.linalg.svd(E_normalized)
        S[0] = (S[0] + S[1]) / 2.0
        S[1] = S[0]
        S[2] = 0
        E = U @ np.diag(S) @ V

        self.params = dst_matrix.T @ E @ src_matrix

        return None

    @_deprecate_estimate
    def estimate(self, src, dst):
        """Estimate essential matrix using 8-point algorithm.

        The 8-point algorithm requires at least 8 corresponding point pairs for
        a well-conditioned solution, otherwise the over-determined solution is
        estimated.

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """
        return self._estimate(src, dst) is None


class ProjectiveTransform(_HMatrixTransform):
    r"""Projective transformation.

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
    matrix : (D+1, D+1) array_like, optional
        Homogeneous transformation matrix.
    dimensionality : int, optional
        Fallback number of dimensions when `matrix` not specified.

    Attributes
    ----------
    params : (D+1, D+1) array
        Homogeneous transformation matrix.

    Examples
    --------
    >>> import numpy as np
    >>> import skimage as ski

    Define a transform with an homogeneous transformation matrix:

    >>> tform = ski.transform.ProjectiveTransform(np.diag([2., 3., 1.]))
    >>> tform.params
    array([[2., 0., 0.],
           [0., 3., 0.],
           [0., 0., 1.]])

    You can estimate a transformation to map between source and destination
    points:

    >>> src = np.array([[150, 150],
    ...                 [250, 100],
    ...                 [150, 200]])
    >>> dst = np.array([[200, 200],
    ...                 [300, 150],
    ...                 [150, 400]])
    >>> tform = ski.transform.ProjectiveTransform.from_estimate(src, dst)
    >>> np.allclose(tform.params, [[ -16.56,    5.82,  895.81],
    ...                            [ -10.31,   -8.29, 2075.43],
    ...                            [  -0.05,    0.02,    1.  ]], atol=0.01)
    True

    Apply the transformation to some image data.

    >>> img = ski.data.astronaut()
    >>> warped = ski.transform.warp(img, inverse_map=tform.inverse)

    The estimation can fail - for example, if all the input or output points
    are the same.  If this happens, you will get a transform that is not
    "truthy" - meaning that ``bool(tform)`` is ``False``:

    >>> # A successfully estimated model is truthy (applying ``bool()``
    >>> # gives ``True``):
    >>> if tform:
    ...     print("Estimation succeeded.")
    Estimation succeeded.
    >>> # Not so for a degenerate transform with identical points.
    >>> bad_src = np.ones((3, 2))
    >>> bad_tform = ski.transform.ProjectiveTransform.from_estimate(
    ...      bad_src, dst)
    >>> if not bad_tform:
    ...     print("Estimation failed.")
    Estimation failed.

    Trying to use this failed estimation transform result will give a suitable
    error:

    >>> bad_tform.params  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      ...
    FailedEstimationAccessError: No attribute "params" for failed estimation ...
    """

    scaling = 'rms'

    @property
    def _coeff_inds(self):
        """Indices into flat ``self.params`` with coefficients to estimate"""
        return range(self.params.size - 1)

    def _check_dims(self, d):
        if d >= 2:
            return
        raise NotImplementedError(
            f'Input for {type(self)} should result in transform of >=2D'
        )

    @property
    def _inv_matrix(self):
        return np.linalg.inv(self.params)

    def __array__(self, dtype=None, copy=None):
        return self.params if dtype is None else self.params.astype(dtype)

    def __call__(self, coords):
        """Apply forward transformation.

        Parameters
        ----------
        coords : (N, D) array_like
            Source coordinates.

        Returns
        -------
        coords_out : (N, D) array
            Destination coordinates.

        """
        return _apply_homogeneous(self.params, coords)

    @property
    def inverse(self):
        """Return a transform object representing the inverse."""
        return type(self)(matrix=self._inv_matrix)

    @classmethod
    def from_estimate(cls, src, dst, weights=None):
        """Estimate the transformation from a set of corresponding points.

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

        Weights can be applied to each pair of corresponding points to
        indicate, particularly in an overdetermined system, if point pairs have
        higher or lower confidence or uncertainties associated with them. From
        the matrix treatment of least squares problems, these weight values are
        normalized, square-rooted, then built into a diagonal matrix, by which
        A is multiplied.

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
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.
        weights : (N,) array_like, optional
            Relative weight values for each pair of points.

        Returns
        -------
        tf : Self or ``FailedEstimation``
            An instance of the transformation if the estimation succeeded.
            Otherwise, we return a special ``FailedEstimation`` object to
            signal a failed estimation. Testing the truth value of the failed
            estimation object will return ``False``. E.g.

            .. code-block:: python

                tf = ProjectiveTransform.from_estimate(...)
                if not tf:
                    raise RuntimeError(f"Failed estimation: {tf}")

        """
        return super().from_estimate(src, dst, weights)

    def _estimate(self, src, dst, weights=None):
        src = np.asarray(src)
        dst = np.asarray(dst)
        n, d = src.shape
        fail_matrix = np.full((d + 1, d + 1), np.nan)

        src_matrix, src = _center_and_normalize_points(src)
        dst_matrix, dst = _center_and_normalize_points(dst)
        if not np.all(np.isfinite(src_matrix + dst_matrix)):
            self.params = fail_matrix
            return 'Scaling generated NaN values'

        # params: a0, a1, a2, b0, b1, b2, c0, c1
        A = np.zeros((n * d, (d + 1) ** 2))
        # fill the A matrix with the appropriate block matrices; see docstring
        # for 2D example — this can be generalised to more blocks in the 3D and
        # higher-dimensional cases.
        for ddim in range(d):
            A[ddim * n : (ddim + 1) * n, ddim * (d + 1) : ddim * (d + 1) + d] = src
            A[ddim * n : (ddim + 1) * n, ddim * (d + 1) + d] = 1
            A[ddim * n : (ddim + 1) * n, -d - 1 : -1] = src
            A[ddim * n : (ddim + 1) * n, -1] = -1
            A[ddim * n : (ddim + 1) * n, -d - 1 :] *= -dst[:, ddim : (ddim + 1)]

        # Select relevant columns, depending on params
        A = A[:, list(self._coeff_inds) + [-1]]

        # Get the vectors that correspond to singular values, also applying
        # the weighting if provided
        if weights is None:
            _, _, V = np.linalg.svd(A)
        else:
            weights = np.asarray(weights)
            W = np.diag(np.tile(np.sqrt(weights / np.max(weights)), d))
            _, _, V = np.linalg.svd(W @ A)

        H = np.zeros((d + 1, d + 1))
        # Solution is right singular vector that corresponds to smallest
        # singular value.
        if np.isclose(V[-1, -1], 0):
            self.params = fail_matrix
            return 'Right singular vector has 0 final element'

        H.flat[list(self._coeff_inds) + [-1]] = -V[-1, :-1] / V[-1, -1]
        H[d, d] = 1

        # De-center and de-normalize
        H = np.linalg.inv(dst_matrix) @ H @ src_matrix

        # Small errors can creep in if points are not exact, causing the last
        # element of H to deviate from unity. Correct for that here.
        H /= H[-1, -1]

        self.params = H

        return None

    def __add__(self, other):
        """Combine this transformation with another."""
        if isinstance(other, ProjectiveTransform):
            # combination of the same types result in a transformation of this
            # type again, otherwise use general projective transformation
            if type(self) == type(other):
                tform = self.__class__
            else:
                tform = ProjectiveTransform
            return tform(other.params @ self.params)
        else:
            raise TypeError("Cannot combine transformations of differing " "types.")

    def __nice__(self):
        """common 'paramstr' used by __str__ and __repr__"""
        if not hasattr(self, 'params'):
            return '<not yet initialized>'
        npstring = np.array2string(self.params, separator=', ')
        return 'matrix=\n' + textwrap.indent(npstring, '    ')

    def __repr__(self):
        """Add standard repr formatting around a __nice__ string"""
        return f'<{type(self).__name__}({self.__nice__()}) at {hex(id(self))}>'

    def __str__(self):
        """Add standard str formatting around a __nice__ string"""
        return f'<{type(self).__name__}({self.__nice__()})>'

    @property
    def dimensionality(self):
        """The dimensionality of the transformation."""
        return self.params.shape[0] - 1

    @classmethod
    def identity(cls, dimensionality=None):
        """Identity transform

        Parameters
        ----------
        dimensionality : {None, int}, optional
            Dimensionality of identity transform.

        Returns
        -------
        tform : transform
            Transform such that ``np.all(tform(pts) == pts)``.
        """
        return super().identity(dimensionality=dimensionality)

    @_deprecate_estimate
    def estimate(self, src, dst, weights=None):
        """Estimate the transformation from a set of corresponding points.

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

        Weights can be applied to each pair of corresponding points to
        indicate, particularly in an overdetermined system, if point pairs have
        higher or lower confidence or uncertainties associated with them. From
        the matrix treatment of least squares problems, these weight values are
        normalized, square-rooted, then built into a diagonal matrix, by which
        A is multiplied.

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
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.
        weights : (N,) array_like, optional
            Relative weight values for each pair of points.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """
        return self._estimate(src, dst, weights) is None


@_update_from_estimate_docstring
@_deprecate_inherited_estimate
class AffineTransform(ProjectiveTransform):
    """Affine transformation.

    Has the following form::

        X = a0 * x + a1 * y + a2
          =   sx * x * [cos(rotation) + tan(shear_y) * sin(rotation)]
            - sy * y * [tan(shear_x) * cos(rotation) + sin(rotation)]
            + translation_x

        Y = b0 * x + b1 * y + b2
          =   sx * x * [sin(rotation) - tan(shear_y) * cos(rotation)]
            - sy * y * [tan(shear_x) * sin(rotation) - cos(rotation)]
            + translation_y

    where ``sx`` and ``sy`` are scale factors in the x and y directions.

    This is equivalent to applying the operations in the following order:

    1. Scale
    2. Shear
    3. Rotate
    4. Translate

    The homogeneous transformation matrix is::

        [[a0  a1  a2]
         [b0  b1  b2]
         [0   0    1]]

    In 2D, the transformation parameters can be given as the homogeneous
    transformation matrix, above, or as the implicit parameters, scale,
    rotation, shear, and translation in x (a2) and y (b2). For 3D and higher,
    only the matrix form is allowed.

    In narrower transforms, such as the Euclidean (only rotation and
    translation) or Similarity (rotation, translation, and a global scale
    factor) transforms, it is possible to specify 3D transforms using implicit
    parameters also.

    Parameters
    ----------
    matrix : (D+1, D+1) array_like, optional
        Homogeneous transformation matrix. If this matrix is provided, it is an
        error to provide any of scale, rotation, shear, or translation.
    scale : {s as float or (sx, sy) as array, list or tuple}, optional
        Scale factor(s). If a single value, it will be assigned to both
        sx and sy. Only available for 2D.

        .. versionadded:: 0.17
           Added support for supplying a single scalar value.
    shear : float or 2-tuple of float, optional
        The x and y shear angles, clockwise, by which these axes are
        rotated around the origin [2].
        If a single value is given, take that to be the x shear angle, with
        the y angle remaining 0. Only available in 2D.
    rotation : float, optional
        Rotation angle, clockwise, as radians. Only available for 2D.
    translation : (tx, ty) as array, list or tuple, optional
        Translation parameters. Only available for 2D.
    dimensionality : int, optional
        Fallback number of dimensions for transform when none of `matrix`,
        `scale`, `rotation`, `shear` or `translation` are specified.  If any of
        `scale`, `rotation`, `shear` or `translation` are specified, must equal
        2 (the default).

    Attributes
    ----------
    params : (D+1, D+1) array
        Homogeneous transformation matrix.

    Raises
    ------
    ValueError
        If both ``matrix`` and any of the other parameters are provided.

    Examples
    --------
    >>> import numpy as np
    >>> import skimage as ski

    Define a transform with an homogeneous transformation matrix:

    >>> tform = ski.transform.AffineTransform(np.diag([2., 3., 1.]))
    >>> tform.params
    array([[2., 0., 0.],
           [0., 3., 0.],
           [0., 0., 1.]])

    Define a transform with parameters:

    >>> tform = ski.transform.AffineTransform(scale=4, rotation=0.2)
    >>> np.round(tform.params, 2)
    array([[ 3.92, -0.79,  0.  ],
           [ 0.79,  3.92,  0.  ],
           [ 0.  ,  0.  ,  1.  ]])

    You can estimate a transformation to map between source and destination
    points:

    >>> src = np.array([[150, 150],
    ...                 [250, 100],
    ...                 [150, 200]])
    >>> dst = np.array([[200, 200],
    ...                 [300, 150],
    ...                 [150, 400]])
    >>> tform = ski.transform.AffineTransform.from_estimate(src, dst)
    >>> np.allclose(tform.params, [[   0.5,   -1. ,  275. ],
    ...                            [   1.5,    4. , -625. ],
    ...                            [   0. ,    0. ,    1. ]])
    True

    Apply the transformation to some image data.

    >>> img = ski.data.astronaut()
    >>> warped = ski.transform.warp(img, inverse_map=tform.inverse)

    The estimation can fail - for example, if all the input or output points
    are the same.  If this happens, you will get a transform that is not
    "truthy" - meaning that ``bool(tform)`` is ``False``:

    >>> # A successfully estimated model is truthy (applying ``bool()``
    >>> # gives ``True``):
    >>> if tform:
    ...     print("Estimation succeeded.")
    Estimation succeeded.
    >>> # Not so for a degenerate transform with identical points.
    >>> bad_src = np.ones((3, 2))
    >>> bad_tform = ski.transform.AffineTransform.from_estimate(
    ...      bad_src, dst)
    >>> if not bad_tform:
    ...     print("Estimation failed.")
    Estimation failed.

    Trying to use this failed estimation transform result will give a suitable
    error:

    >>> bad_tform.params  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      ...
    FailedEstimationAccessError: No attribute "params" for failed estimation ...

    References
    ----------
    .. [1] Wikipedia, "Affine transformation",
           https://en.wikipedia.org/wiki/Affine_transformation#Image_transformation
    .. [2] Wikipedia, "Shear mapping",
           https://en.wikipedia.org/wiki/Shear_mapping
    """

    def __init__(
        self,
        matrix=None,
        *,
        scale=None,
        shear=None,
        rotation=None,
        translation=None,
        dimensionality=None,
    ):
        n_srst_none = sum(p is None for p in (scale, rotation, shear, translation))
        if n_srst_none != 4:
            if matrix is not None:
                raise ValueError(
                    "Do not specify any implicit parameters when "
                    "matrix is specified."
                )
            if dimensionality is not None and dimensionality > 2:
                raise ValueError('Implicit parameters only valid for 2D transforms')
            # 2D parameter checks explicit or implicit in _srst2matrix.
            matrix = self._srst2matrix(scale, rotation, shear, translation)
            if matrix.shape[0] != 3:
                raise ValueError('Implicit parameters must give 2D transforms')
        super().__init__(matrix=matrix, dimensionality=dimensionality)

    @property
    def _coeff_inds(self):
        """Indices into flat ``self.params`` with coefficients to estimate"""
        return range(self.dimensionality * (self.dimensionality + 1))

    def _srst2matrix(self, scale, rotation, shear, translation):
        scale = (1, 1) if scale is None else scale
        sx, sy = (scale, scale) if np.isscalar(scale) else scale
        rotation = 0 if rotation is None else rotation
        if not np.isscalar(rotation):
            raise ValueError('rotation must be scalar (2D rotation)')
        shear = 0 if shear is None else shear
        shear_x, shear_y = (shear, 0) if np.isscalar(shear) else shear
        translation = (0, 0) if translation is None else translation
        if np.isscalar(translation):
            raise ValueError('translation must be length 2')
        a2, b2 = translation

        a0 = sx * (math.cos(rotation) + math.tan(shear_y) * math.sin(rotation))
        a1 = -sy * (math.tan(shear_x) * math.cos(rotation) + math.sin(rotation))

        b0 = sx * (math.sin(rotation) - math.tan(shear_y) * math.cos(rotation))
        b1 = -sy * (math.tan(shear_x) * math.sin(rotation) - math.cos(rotation))
        return np.array([[a0, a1, a2], [b0, b1, b2], [0, 0, 1]])

    @property
    def scale(self):
        if self.dimensionality != 2:
            return np.sqrt(np.sum(self.params**2, axis=0))[: self.dimensionality]
        ss = np.sum(self.params**2, axis=0)
        ss[1] = ss[1] / (math.tan(self.shear) ** 2 + 1)
        return np.sqrt(ss)[: self.dimensionality]

    @property
    def rotation(self):
        if self.dimensionality != 2:
            raise NotImplementedError(
                'The rotation property is only implemented for 2D transforms.'
            )
        return math.atan2(self.params[1, 0], self.params[0, 0])

    @property
    def shear(self):
        if self.dimensionality != 2:
            raise NotImplementedError(
                'The shear property is only implemented for 2D transforms.'
            )
        beta = math.atan2(-self.params[0, 1], self.params[1, 1])
        return beta - self.rotation

    @property
    def translation(self):
        return self.params[0 : self.dimensionality, self.dimensionality]


class PiecewiseAffineTransform(_GeometricTransform):
    """Piecewise affine transformation.

    Control points are used to define the mapping. The transform is based on
    a Delaunay triangulation of the points to form a mesh. Each triangle is
    used to find a local affine transform.

    Attributes
    ----------
    affines : list of AffineTransform objects
        Affine transformations for each triangle in the mesh.
    inverse_affines : list of AffineTransform objects
        Inverse affine transformations for each triangle in the mesh.

    Examples
    --------
    >>> import numpy as np
    >>> import skimage as ski

    Define a transformation by estimation:

    >>> src = [[-12.3705, -10.5075],
    ...        [-10.7865, 15.4305],
    ...        [8.6985, 10.8675],
    ...        [11.4975, -9.5715],
    ...        [7.8435, 7.4835],
    ...        [-5.3325, 6.5025],
    ...        [6.7905, -6.3765],
    ...        [-6.1695, -0.8235]]
    >>> dst = [[0, 0],
    ...        [0, 5800],
    ...        [4900, 5800],
    ...        [4900, 0],
    ...        [4479, 4580],
    ...        [1176, 3660],
    ...        [3754, 790],
    ...        [1024, 1931]]
    >>> tform = ski.transform.PiecewiseAffineTransform.from_estimate(src, dst)

    Calling the transform applies the transformation to the points:

    >>> np.allclose(tform(src), dst)
    True

    You can apply the inverse transform:

    >>> np.allclose(tform.inverse(dst), src)
    True

    The estimation can fail - for example, if all the input or output points
    are the same.  If this happens, you will get a transform that is not
    "truthy" - meaning that ``bool(tform)`` is ``False``:

    >>> # A successfully estimated model is truthy (applying ``bool()``
    >>> # gives ``True``):
    >>> if tform:
    ...     print("Estimation succeeded.")
    Estimation succeeded.
    >>> # Not so for a degenerate transform with identical points.
    >>> bad_src = [[1, 1]] * 6 + src[6:]
    >>> bad_tform = ski.transform.PiecewiseAffineTransform.from_estimate(
    ...      bad_src, dst)
    >>> if not bad_tform:
    ...     print("Estimation failed.")
    Estimation failed.

    Trying to use this failed estimation transform result will give a suitable
    error:

    >>> bad_tform.params  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      ...
    FailedEstimationAccessError: No attribute "params" for failed estimation ...
    """

    def __init__(self):
        self._tesselation = None
        self._inverse_tesselation = None
        self.affines = None
        self.inverse_affines = None

    @classmethod
    def from_estimate(cls, src, dst):
        """Estimate the transformation from a set of corresponding points.

        Number of source and destination coordinates must match.

        Parameters
        ----------
        src : (N, D) array_like
            Source coordinates.
        dst : (N, D) array_like
            Destination coordinates.

        Returns
        -------
        tf : Self or ``FailedEstimation``
            An instance of the transformation if the estimation succeeded.
            Otherwise, we return a special ``FailedEstimation`` object to
            signal a failed estimation. Testing the truth value of the failed
            estimation object will return ``False``. E.g.

            .. code-block:: python

                tf = PiecewiseAffineTransform.from_estimate(...)
                if not tf:
                    raise RuntimeError(f"Failed estimation: {tf}")

        """
        return super().from_estimate(src, dst)

    def _estimate(self, src, dst):
        src = np.asarray(src)
        dst = np.asarray(dst)
        N, D = src.shape

        # forward piecewise affine
        # triangulate input positions into mesh
        self._tesselation = spatial.Delaunay(src)

        fail_matrix = np.full((D + 1, D + 1), np.nan)

        # find affine mapping from source positions to destination
        self.affines = []
        messages = []
        for i, tri in enumerate(self._tesselation.simplices):
            affine = AffineTransform.from_estimate(src[tri, :], dst[tri, :])
            if not affine:
                messages.append(f'Failure at forward simplex {i}: {affine}')
                affine = AffineTransform(fail_matrix.copy())
            self.affines.append(affine)

        # inverse piecewise affine
        # triangulate input positions into mesh
        self._inverse_tesselation = spatial.Delaunay(dst)
        # find affine mapping from source positions to destination
        self.inverse_affines = []
        for i, tri in enumerate(self._inverse_tesselation.simplices):
            affine = AffineTransform.from_estimate(dst[tri, :], src[tri, :])
            if not affine:
                messages.append(f'Failure at inverse simplex {i}: {affine}')
                affine = AffineTransform(fail_matrix.copy())
            self.inverse_affines.append(affine)

        return '; '.join(messages) if messages else None

    def __call__(self, coords):
        """Apply forward transformation.

        Coordinates outside of the mesh will be set to `- 1`.

        Parameters
        ----------
        coords : (N, D) array_like
            Source coordinates.

        Returns
        -------
        coords : (N, 2) array
            Transformed coordinates.

        """
        coords = np.asarray(coords)
        out = np.empty_like(coords, np.float64)

        # determine triangle index for each coordinate
        simplex = self._tesselation.find_simplex(coords)

        # coordinates outside of mesh
        out[simplex == -1, :] = -1

        for index in range(len(self._tesselation.simplices)):
            # affine transform for triangle
            affine = self.affines[index]
            # all coordinates within triangle
            index_mask = simplex == index

            out[index_mask, :] = affine(coords[index_mask, :])

        return out

    @property
    def inverse(self):
        """Return a transform object representing the inverse."""
        tform = type(self)()
        # Copy parameters (None or list) for safety.
        tform._tesselation = copy(self._inverse_tesselation)
        tform._inverse_tesselation = copy(self._tesselation)
        tform.affines = copy(self.inverse_affines)
        tform.inverse_affines = copy(self.affines)
        return tform

    @classmethod
    def identity(cls, dimensionality=None):
        """Identity transform

        Parameters
        ----------
        dimensionality : optional
            This transform does not use the `dimensionality` parameter, so the
            value is ignored.  The parameter exists for compatibility with
            other transforms.

        Returns
        -------
        tform : transform
            Transform such that ``np.all(tform(pts) == pts)``.
        """
        return cls()

    @_deprecate_estimate
    def estimate(self, src, dst):
        """Estimate the transformation from a set of corresponding points.

        Number of source and destination coordinates must match.

        Parameters
        ----------
        src : (N, D) array_like
            Source coordinates.
        dst : (N, D) array_like
            Destination coordinates.

        Returns
        -------
        success : bool
            True, if all pieces of the model are successfully estimated.

        """
        return self._estimate(src, dst) is None


def _euler_rotation_matrix(angles, degrees=False):
    """Produce an Euler rotation matrix from the given intrinsic rotation angles
    for the axes x, y and z.

    Parameters
    ----------
    angles : array of float, shape (3,)
        The transformation angles in radians.
    degrees : bool, optional
        If True, then the given angles are assumed to be in degrees. Default is False.

    Returns
    -------
    R : array of float, shape (3, 3)
        The Euler rotation matrix.

    """
    return spatial.transform.Rotation.from_euler(
        'XYZ', angles=angles, degrees=degrees
    ).as_matrix()


class EuclideanTransform(ProjectiveTransform):
    """Euclidean transformation, also known as a rigid transform.

    Has the following form::

        X = a0 * x - b0 * y + a1 =
          = x * cos(rotation) - y * sin(rotation) + a1

        Y = b0 * x + a0 * y + b1 =
          = x * sin(rotation) + y * cos(rotation) + b1

    where the homogeneous transformation matrix is::

        [[a0 -b0  a1]
         [b0  a0  b1]
         [0   0   1 ]]

    The Euclidean transformation is a rigid transformation with rotation and
    translation parameters. The similarity transformation extends the Euclidean
    transformation with a single scaling factor.

    In 2D and 3D, the transformation parameters may be provided either via
    `matrix`, the homogeneous transformation matrix, above, or via the
    implicit parameters `rotation` and/or `translation` (where `a1` is the
    translation along `x`, `b1` along `y`, etc.). Beyond 3D, if the
    transformation is only a translation, you may use the implicit parameter
    `translation`; otherwise, you must use `matrix`.

    The implicit parameters are applied in the following order:

    1. Rotation;
    2. Translation.

    Parameters
    ----------
    matrix : (D+1, D+1) array_like, optional
        Homogeneous transformation matrix.
    rotation : float or sequence of float, optional
        Rotation angle, clockwise, in radians. If given as a vector, it is
        interpreted as Euler rotation angles [1]_. Only 2D (single rotation)
        and 3D (Euler rotations) values are supported. For higher dimensions,
        you must provide or estimate the transformation matrix instead, and
        pass that as `matrix` above.
    translation : (x, y[, z, ...]) sequence of float, length D, optional
        Translation parameters for each axis.
    dimensionality : int, optional
        Fallback number of dimensions for transform when no other parameter
        is specified.  Otherwise ignored, and we infer dimensionality from the
        input parameters.

    Attributes
    ----------
    params : (D+1, D+1) array
        Homogeneous transformation matrix.

    Examples
    --------
    >>> import numpy as np
    >>> import skimage as ski

    Define a transform with an homogeneous transformation matrix:

    >>> tform = ski.transform.EuclideanTransform(np.diag([2., 3., 1.]))
    >>> tform.params
    array([[2., 0., 0.],
           [0., 3., 0.],
           [0., 0., 1.]])

    Define a transform with parameters:

    >>> tform = ski.transform.EuclideanTransform(
    ...             rotation=0.2, translation=[1, 2])
    >>> np.round(tform.params, 2)
    array([[ 0.98, -0.2 ,  1.  ],
           [ 0.2 ,  0.98,  2.  ],
           [ 0.  ,  0.  ,  1.  ]])

    You can estimate a transformation to map between source and destination
    points:

    >>> src = np.array([[150, 150],
    ...                 [250, 100],
    ...                 [150, 200]])
    >>> dst = np.array([[200, 200],
    ...                 [300, 150],
    ...                 [150, 400]])
    >>> tform = ski.transform.EuclideanTransform.from_estimate(src, dst)
    >>> np.allclose(tform.params, [[ 0.99, 0.12,  16.77],
    ...                            [-0.12, 0.99, 122.91],
    ...                            [ 0.  , 0.  ,   1.  ]], atol=0.01)
    True

    Apply the transformation to some image data.

    >>> img = ski.data.astronaut()
    >>> warped = ski.transform.warp(img, inverse_map=tform.inverse)

    The estimation can fail - for example, if all the input or output points
    are the same.  If this happens, you will get a transform that is not
    "truthy" - meaning that ``bool(tform)`` is ``False``:

    >>> # A successfully estimated model is truthy (applying ``bool()``
    >>> # gives ``True``):
    >>> if tform:
    ...     print("Estimation succeeded.")
    Estimation succeeded.
    >>> # Not so for a degenerate transform with identical points.
    >>> bad_src = np.ones((3, 2))
    >>> bad_tform = ski.transform.EuclideanTransform.from_estimate(
    ...      bad_src, dst)
    >>> if not bad_tform:
    ...     print("Estimation failed.")
    Estimation failed.

    Trying to use this failed estimation transform result will give a suitable
    error:

    >>> bad_tform.params  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      ...
    FailedEstimationAccessError: No attribute "params" for failed estimation ...

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions

    """

    # Whether to estimate scale during estimation.
    _estimate_scale = False

    def __init__(
        self, matrix=None, *, rotation=None, translation=None, dimensionality=None
    ):
        n_rt_none = sum(p is None for p in (rotation, translation))
        if n_rt_none != 2:
            if matrix is not None:
                raise ValueError(
                    "Do not specify any implicit parameters when "
                    "matrix is specified."
                )
            n_dims, chk_msg = self._rt2ndims_msg(rotation, translation)
            if chk_msg is not None:
                raise ValueError(chk_msg)
            matrix = self._rt2matrix(rotation, translation, n_dims)
        super().__init__(matrix=matrix, dimensionality=dimensionality)

    def _rt2ndims_msg(self, rotation, translation):
        if rotation is not None:
            N = 1 if np.isscalar(rotation) else len(rotation)
            msg = (
                '``rotations`` must be scalar (3D) or length 3 (3D)'
                if N not in (1, 3)
                else None
            )
            return 2 if N == 1 else N, msg
        if translation is not None:
            return (2 if np.isscalar(translation) else len(translation), None)
        return None, None

    def _rt2matrix(self, rotation, translation, n_dims):
        if translation is None:
            translation = (0,) * n_dims
        if rotation is None:
            rotation = 0 if n_dims == 2 else np.zeros(3)
        matrix = np.eye(n_dims + 1)
        if n_dims == 2:
            cos_r, sin_r = math.cos(rotation), math.sin(rotation)
            matrix[:2, :2] = [[cos_r, -sin_r], [sin_r, cos_r]]
        elif n_dims == 3:
            matrix[:3, :3] = _euler_rotation_matrix(rotation)
        matrix[0:n_dims, n_dims] = translation
        return matrix

    @classmethod
    def from_estimate(cls, src, dst) -> Self | FailedEstimation:
        """Estimate the transformation from a set of corresponding points.

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.

        Returns
        -------
        tf : Self or ``FailedEstimation``
            An instance of the transformation if the estimation succeeded.
            Otherwise, we return a special ``FailedEstimation`` object to
            signal a failed estimation. Testing the truth value of the failed
            estimation object will return ``False``. E.g.

            .. code-block:: python

                tf = EuclideanTransform.from_estimate(...)
                if not tf:
                    raise RuntimeError(f"Failed estimation: {tf}")

        """
        # Use base implementation to avoid weights argument of
        # ProjectiveTransform ancestor class.
        return _from_estimate(cls, src, dst)

    def _estimate(self, src, dst):
        self.params = _umeyama(src, dst, self._estimate_scale)

        # _umeyama will return nan if the problem is not well-conditioned.
        return (
            'Poor conditioning for estimation'
            if np.any(np.isnan(self.params))
            else None
        )

    @property
    def rotation(self):
        if self.dimensionality == 2:
            return math.atan2(self.params[1, 0], self.params[1, 1])
        elif self.dimensionality == 3:
            # Returning 3D Euler rotation matrix
            return self.params[:3, :3]
        else:
            raise NotImplementedError(
                'Rotation only implemented for 2D and 3D transforms.'
            )

    @property
    def translation(self):
        return self.params[0 : self.dimensionality, self.dimensionality]

    @_deprecate_estimate
    def estimate(self, src, dst):
        """Estimate the transformation from a set of corresponding points.

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """
        return self._estimate(src, dst) is None


@_update_from_estimate_docstring
@_deprecate_inherited_estimate
class SimilarityTransform(EuclideanTransform):
    """Similarity transformation.

    Has the following form in 2D::

        X = a0 * x - b0 * y + a1 =
          = s * x * cos(rotation) - s * y * sin(rotation) + a1

        Y = b0 * x + a0 * y + b1 =
          = s * x * sin(rotation) + s * y * cos(rotation) + b1

    where ``s`` is a scale factor and the homogeneous transformation matrix is::

        [[a0 -b0  a1]
         [b0  a0  b1]
         [0   0   1 ]]

    The similarity transformation extends the Euclidean transformation with a
    single scaling factor in addition to the rotation and translation
    parameters.

    The implicit parameters are applied in the following order:

    1. Scale;
    2. Rotation;
    3. Translation.

    Parameters
    ----------
    matrix : (dim+1, dim+1) array_like, optional
        Homogeneous transformation matrix.
    scale : float, optional
        Scale factor. Implemented only for 2D and 3D.
    rotation : float, optional
        Rotation angle, clockwise, as radians.
        Implemented only for 2D and 3D. For 3D, this is given in ZYX Euler
        angles.
    translation : (dim,) array_like, optional
        x, y[, z] translation parameters. Implemented only for 2D and 3D.
    dimensionality : int, optional
        The dimensionality of the transform, corresponding to ``dim`` above.
        Ignored if `matrix` is not None, and set to ``matrix.shape[0] - 1``.
        Otherwise, must be one of 2 or 3.

    Attributes
    ----------
    params : (dim+1, dim+1) array
        Homogeneous transformation matrix.

    Examples
    --------
    >>> import numpy as np
    >>> import skimage as ski

    Define a transform with an homogeneous transformation matrix:

    >>> tform = ski.transform.SimilarityTransform(np.diag([2., 3., 1.]))
    >>> tform.params
    array([[2., 0., 0.],
           [0., 3., 0.],
           [0., 0., 1.]])

    Define a transform with parameters:

    >>> tform = ski.transform.SimilarityTransform(
    ...             rotation=0.2, translation=[1, 2])
    >>> np.round(tform.params, 2)
    array([[ 0.98, -0.2 ,  1.  ],
           [ 0.2 ,  0.98,  2.  ],
           [ 0.  ,  0.  ,  1.  ]])

    You can estimate a transformation to map between source and destination
    points:

    >>> src = np.array([[150, 150],
    ...                 [250, 100],
    ...                 [150, 200]])
    >>> dst = np.array([[200, 200],
    ...                 [300, 150],
    ...                 [150, 400]])
    >>> tform = ski.transform.SimilarityTransform.from_estimate(src, dst)
    >>> np.allclose(tform.params, [[ 1.79, 0.21, -142.86],
    ...                            [-0.21, 1.79,   21.43],
    ...                            [ 0.  , 0.  ,    1.  ]], atol=0.01)
    True

    Apply the transformation to some image data.

    >>> img = ski.data.astronaut()
    >>> warped = ski.transform.warp(img, inverse_map=tform.inverse)

    The estimation can fail - for example, if all the input or output points
    are the same.  If this happens, you will get a transform that is not
    "truthy" - meaning that ``bool(tform)`` is ``False``:

    >>> # A successfully estimated model is truthy (applying ``bool()``
    >>> # gives ``True``):
    >>> if tform:
    ...     print("Estimation succeeded.")
    Estimation succeeded.
    >>> # Not so for a degenerate transform with identical points.
    >>> bad_src = np.ones((3, 2))
    >>> bad_tform = ski.transform.SimilarityTransform.from_estimate(
    ...      bad_src, dst)
    >>> if not bad_tform:
    ...     print("Estimation failed.")
    Estimation failed.

    Trying to use this failed estimation transform result will give a suitable
    error:

    >>> bad_tform.params  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      ...
    FailedEstimationAccessError: No attribute "params" for failed estimation ...
    """

    # Whether to estimate scale during estimation.
    _estimate_scale = True

    def __init__(
        self,
        matrix=None,
        *,
        scale=None,
        rotation=None,
        translation=None,
        dimensionality=None,
    ):
        n_srt_none = sum(p is None for p in (scale, rotation, translation))
        if n_srt_none != 3:
            if matrix is not None:
                raise ValueError(
                    "Do not specify any implicit parameters when "
                    "matrix is specified."
                )
            self._check_scale(scale, (rotation, translation), dimensionality)
            # Scale is special.  Scalar scale does not tell us the dimensions.
            if scale is not None and not np.isscalar(scale):
                n_dims, chk_msg = len(scale), None
            else:
                n_dims, chk_msg = self._rt2ndims_msg(rotation, translation)
            if chk_msg is not None:
                raise ValueError(chk_msg)
            # n_dims can be None for scalar scale, other parameters are None.
            n_dims = (
                n_dims
                if n_dims is not None
                else dimensionality
                if dimensionality is not None
                else 2
            )
            matrix = self._rt2matrix(rotation, translation, n_dims)
            if scale not in (None, 1):
                matrix[:n_dims, :n_dims] *= scale
        super().__init__(matrix=matrix, dimensionality=dimensionality)

    def _check_scale(self, scale, other_params, dimensionality):
        """Check, warn for scalar scaling"""
        if dimensionality in (None, 2) or scale is None or not np.isscalar(scale):
            return
        if all(p is None for p in other_params):
            warnings.warn(
                'In the future, it will be a ValueError to pass a '
                'scalar `scale` value with a ``dimensionality`` '
                '> 2\n,and without other implicit parameters '
                'to indicate the dimensionality of the transform.\n'
                'Please indicate dimensionality by passing a vector '
                'of suitable length to `scale`.',
                FutureWarning,
                stacklevel=2,
            )

    @property
    def scale(self):
        # det = scale**(# of dimensions), therefore scale = det**(1/ndim)
        if self.dimensionality == 2:
            return np.sqrt(np.linalg.det(self.params))
        elif self.dimensionality == 3:
            return np.cbrt(np.linalg.det(self.params))
        else:
            raise NotImplementedError('Scale is only implemented for 2D and 3D.')


class PolynomialTransform(_GeometricTransform):
    """2D polynomial transformation.

    Has the following form::

        X = sum[j=0:order]( sum[i=0:j]( a_ji * x**(j - i) * y**i ))
        Y = sum[j=0:order]( sum[i=0:j]( b_ji * x**(j - i) * y**i ))

    Parameters
    ----------
    params : (2, N) array_like, optional
        Polynomial coefficients where `N * 2 = (order + 1) * (order + 2)`. So,
        a_ji is defined in `params[0, :]` and b_ji in `params[1, :]`.
    dimensionality : int, optional
        Must have value 2 (the default) for polynomial transforms.

    Attributes
    ----------
    params : (2, N) array
        Polynomial coefficients where `N * 2 = (order + 1) * (order + 2)`. So,
        a_ji is defined in `params[0, :]` and b_ji in `params[1, :]`.

    Examples
    --------
    >>> import numpy as np
    >>> import skimage as ski

    Define a transformation by estimation:

    >>> src = [[-12.3705, -10.5075],
    ...        [-10.7865, 15.4305],
    ...        [8.6985, 10.8675],
    ...        [11.4975, -9.5715],
    ...        [7.8435, 7.4835],
    ...        [-5.3325, 6.5025],
    ...        [6.7905, -6.3765],
    ...        [-6.1695, -0.8235]]
    >>> dst = [[0, 0],
    ...        [0, 5800],
    ...        [4900, 5800],
    ...        [4900, 0],
    ...        [4479, 4580],
    ...        [1176, 3660],
    ...        [3754, 790],
    ...        [1024, 1931]]
    >>> tform = ski.transform.PolynomialTransform.from_estimate(src, dst)

    Calling the transform applies the transformation to the points:

    >>> pts = tform(src)
    >>> np.allclose(pts, [[   7.54,   12.27],
    ...                   [   2.98, 5796.95],
    ...                   [4870.44, 5766.59],
    ...                   [4889.72,   -6.72],
    ...                   [4515.62, 4617.5 ],
    ...                   [1183.25, 3694.  ],
    ...                   [3767.57,  800.53],
    ...                   [ 998.02, 1881.97]], atol=0.01)
    True
    """

    def __init__(self, params=None, *, dimensionality=None):
        if dimensionality is None:
            dimensionality = 2
        elif dimensionality != 2:
            raise NotImplementedError(
                'Polynomial transforms are only implemented for 2D.'
            )
        self.params = np.array([[0, 1, 0], [0, 0, 1]] if params is None else params)
        if self.params.shape == () or self.params.shape[0] != 2:
            raise ValueError("Transformation parameters must be shape (2, N)")

    @classmethod
    def from_estimate(cls, src, dst, order=2, weights=None):
        """Estimate the transformation from a set of corresponding points.

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

        Weights can be applied to each pair of corresponding points to
        indicate, particularly in an overdetermined system, if point pairs have
        higher or lower confidence or uncertainties associated with them. From
        the matrix treatment of least squares problems, these weight values are
        normalized, square-rooted, then built into a diagonal matrix, by which
        A is multiplied.

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.
        order : int, optional
            Polynomial order (number of coefficients is order + 1).
        weights : (N,) array_like, optional
            Relative weight values for each pair of points.

        Returns
        -------
        tf : Self or ``FailedEstimation``
            An instance of the transformation if the estimation succeeded.
            Otherwise, we return a special ``FailedEstimation`` object to
            signal a failed estimation. Testing the truth value of the failed
            estimation object will return ``False``. E.g.

            .. code-block:: python

                tf = PolynomialTransform.from_estimate(...)
                if not tf:
                    raise RuntimeError(f"Failed estimation: {tf}")

        """
        return super().from_estimate(src, dst, order, weights)

    def _estimate(self, src, dst, order=2, weights=None):
        src = np.asarray(src)
        dst = np.asarray(dst)
        xs = src[:, 0]
        ys = src[:, 1]
        xd = dst[:, 0]
        yd = dst[:, 1]
        rows = src.shape[0]

        # number of unknown polynomial coefficients
        order = safe_as_int(order)
        u = (order + 1) * (order + 2)

        A = np.zeros((rows * 2, u + 1))
        pidx = 0
        for j in range(order + 1):
            for i in range(j + 1):
                A[:rows, pidx] = xs ** (j - i) * ys**i
                A[rows:, pidx + u // 2] = xs ** (j - i) * ys**i
                pidx += 1

        A[:rows, -1] = xd
        A[rows:, -1] = yd

        # Get the vectors that correspond to singular values, also applying
        # the weighting if provided
        if weights is None:
            _, _, V = np.linalg.svd(A)
        else:
            weights = np.asarray(weights)
            W = np.diag(np.tile(np.sqrt(weights / np.max(weights)), 2))
            _, _, V = np.linalg.svd(W @ A)

        # solution is right singular vector that corresponds to smallest
        # singular value
        params = -V[-1, :-1] / V[-1, -1]

        self.params = params.reshape((2, u // 2))

        return None

    def __call__(self, coords):
        """Apply forward transformation.

        Parameters
        ----------
        coords : (N, 2) array_like
            source coordinates

        Returns
        -------
        coords : (N, 2) array
            Transformed coordinates.

        """
        coords = np.asarray(coords)
        x = coords[:, 0]
        y = coords[:, 1]
        u = len(self.params.ravel())
        # number of coefficients -> u = (order + 1) * (order + 2)
        order = int((-3 + math.sqrt(9 - 4 * (2 - u))) / 2)
        dst = np.zeros(coords.shape)

        pidx = 0
        for j in range(order + 1):
            for i in range(j + 1):
                dst[:, 0] += self.params[0, pidx] * x ** (j - i) * y**i
                dst[:, 1] += self.params[1, pidx] * x ** (j - i) * y**i
                pidx += 1

        return dst

    @classmethod
    def identity(cls, dimensionality=None):
        """Identity transform

        Parameters
        ----------
        dimensionality : {None, 2}, optional
            This transform only allows dimensionality of 2, where None
            corresponds to 2. The parameter exists for compatibility with other
            transforms.

        Returns
        -------
        tform : transform
            Transform such that ``np.all(tform(pts) == pts)``.
        """
        return cls(params=None, dimensionality=dimensionality)

    @property
    def inverse(self):
        raise NotImplementedError(
            'There is no explicit way to do the inverse polynomial '
            'transformation. Instead, estimate the inverse transformation '
            'parameters by exchanging source and destination coordinates,'
            'then apply the forward transformation.'
        )

    @_deprecate_estimate
    def estimate(self, src, dst, order=2, weights=None):
        """Estimate the transformation from a set of corresponding points.

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

        Weights can be applied to each pair of corresponding points to
        indicate, particularly in an overdetermined system, if point pairs have
        higher or lower confidence or uncertainties associated with them. From
        the matrix treatment of least squares problems, these weight values are
        normalized, square-rooted, then built into a diagonal matrix, by which
        A is multiplied.

        Parameters
        ----------
        src : (N, 2) array_like
            Source coordinates.
        dst : (N, 2) array_like
            Destination coordinates.
        order : int, optional
            Polynomial order (number of coefficients is order + 1).
        weights : (N,) array_like, optional
            Relative weight values for each pair of points.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """
        return self._estimate(src, dst, order, weights) is None


TRANSFORMS = {
    'euclidean': EuclideanTransform,
    'similarity': SimilarityTransform,
    'affine': AffineTransform,
    'piecewise-affine': PiecewiseAffineTransform,
    'projective': ProjectiveTransform,
    'fundamental': FundamentalMatrixTransform,
    'essential': EssentialMatrixTransform,
    'polynomial': PolynomialTransform,
}


def estimate_transform(ttype, src, dst, *args, **kwargs):
    """Estimate 2D geometric transformation parameters.

    You can determine the over-, well- and under-determined parameters
    with the total least-squares method.

    Number of source and destination coordinates must match.

    Parameters
    ----------
    ttype : {'euclidean', similarity', 'affine', 'piecewise-affine', \
             'projective', 'polynomial'}
        Type of transform.
    kwargs : array_like or int
        Function parameters (src, dst, n, angle)::

            NAME / TTYPE        FUNCTION PARAMETERS
            'euclidean'         `src, `dst`
            'similarity'        `src, `dst`
            'affine'            `src, `dst`
            'piecewise-affine'  `src, `dst`
            'projective'        `src, `dst`
            'polynomial'        `src, `dst`, `order` (polynomial order,
                                                      default order is 2)

        Also see examples below.

    Returns
    -------
    tf : :class:`_GeometricTransform` or ``FailedEstimation``
        An instance of the requested transformation if the estimation
        Otherwise, we return a special ``FailedEstimation`` object to signal a
        failed estimation. Testing the truth value of the failed estimation
        object will return ``False``. E.g.

        .. code-block:: python

            tf = estimate_transform(...)
            if not tf:
                raise RuntimeError(f"Failed estimation: {tf}")

    Examples
    --------
    >>> import numpy as np
    >>> import skimage as ski

    >>> # estimate transformation parameters
    >>> src = np.array([0, 0, 10, 10]).reshape((2, 2))
    >>> dst = np.array([12, 14, 1, -20]).reshape((2, 2))

    >>> tform = ski.transform.estimate_transform('similarity', src, dst)

    >>> np.allclose(tform.inverse(tform(src)), src)
    True

    >>> # warp image using the estimated transformation
    >>> image = ski.data.camera()

    >>> ski.transform.warp(image, inverse_map=tform.inverse) # doctest: +SKIP

    >>> # create transformation with explicit parameters
    >>> tform2 = ski.transform.SimilarityTransform(scale=1.1, rotation=1,
    ...     translation=(10, 20))

    >>> # unite transformations, applied in order from left to right
    >>> tform3 = tform + tform2
    >>> np.allclose(tform3(src), tform2(tform(src)))
    True

    The estimation can fail - for example, if all the input or output points
    are the same.  If this happens, you will get a transform that is not
    "truthy" - meaning that ``bool(tform)`` is ``False``:

    >>> # A successfully estimated model is truthy (applying ``bool()``
    >>> # gives ``True``):
    >>> if tform:
    ...     print("Estimation succeeded.")
    Estimation succeeded.
    >>> # Not so for a degenerate transform with identical points.
    >>> bad_src = np.ones((2, 2))
    >>> bad_tform = ski.transform.estimate_transform('similarity',
    ...                                              bad_src, dst)
    >>> if not bad_tform:
    ...     print("Estimation failed.")
    Estimation failed.

    Trying to use this failed estimation transform result will give a suitable
    error:

    >>> bad_tform.params  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      ...
    FailedEstimationAccessError: No attribute "params" for failed estimation ...

    """
    ttype = ttype.lower()
    if ttype not in TRANSFORMS:
        raise ValueError(f'the transformation type \'{ttype}\' is not implemented')

    return TRANSFORMS[ttype].from_estimate(src, dst, *args, **kwargs)


def matrix_transform(coords, matrix):
    """Apply 2D matrix transform.

    Parameters
    ----------
    coords : (N, 2) array_like
        x, y coordinates to transform
    matrix : (3, 3) array_like
        Homogeneous transformation matrix.

    Returns
    -------
    coords : (N, 2) array
        Transformed coordinates.

    """
    return ProjectiveTransform(matrix)(coords)
