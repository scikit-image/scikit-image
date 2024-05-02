from itertools import product, combinations_with_replacement
from functools import partial
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from skimage.metrics import normalized_mutual_information
from skimage.transform import pyramid_gaussian
from math import log, floor

"""Parametric image registration

TODO: handle other parametric motion: translation, rotation, etc
TODO: handle color images? channel_axis?
TODO: add other solvers
TODO: merge with PR #3544 https://github.com/seanbudd/scikit-image and #7050 https://github.com/Coilm/scikit-image
TODO: add more test
"""


def _coarse_to_fine_parametric(
    I0,
    I1,
    weights,
    solver,
    pyramid_downscale=2,
    pyramid_minimum_size=16,
    matrix=None,
    dtype=np.float32,
):
    """Coarse to fine solver for affine motion.

    Parameters
    ----------
    I0 : ndarray
        The first gray scale image of the sequence.
    I1 : ndarray
        The second gray scale image of the sequence.
    weights : ndarray
        Weight array the same shape as I0
    solver : callable
        The solver applied at each pyramid level I0,I1,A,v.
    pyramid_downscale : float
        The pyramid downscale factor.
    nlevel : int
        The maximum number of pyramid levels.
    min_size : int
        The minimum size for any dimension of the pyramid levels.
    matrix: np.ndarray
        The initial transformation matrix
    dtype : dtype
        Output data type.

    Returns
    -------
    matrix : ndarray
        The estimated transformation matrix

    Note
    ----
    This function follows skimage.registration._optical_flow_utils.coarse_to_fine

    Generate image pyramids and apply the solver at each level passing the
    previous affine parameters from the previous estimate.

    """

    ndim = I0.ndim

    max_layer = int(
        floor(
            log(min(I0.shape), pyramid_downscale)
            - log(pyramid_minimum_size, pyramid_downscale)
        )
    )

    pyramid = list(
        zip(
            *[
                reversed(
                    list(
                        pyramid_gaussian(
                            x.astype(dtype),
                            max_layer=max_layer,
                            downscale=pyramid_downscale,
                            preserve_range=True,
                        )
                    )
                )
                for x in [I0, I1, weights]
            ]
        )
    )

    if matrix is None:
        matrix = np.eye(ndim + 1)

    matrix = solver(pyramid[0][0], pyramid[0][1], pyramid[0][2], matrix=matrix)

    for J0, J1, W in pyramid[1:]:
        scaled_matrix = matrix
        scaled_matrix[:ndim, -1] = pyramid_downscale * scaled_matrix[:ndim, -1]
        matrix = solver(J0, J1, W, matrix=scaled_matrix)

    return matrix


def _parametric_ilk_solver(
    reference_image, moving_image, weights, num_warp, tol, matrix
):
    """Estimate global affine motion between two images

    Parameters
    ----------

    reference_image : ndarray
        The first gray scale image of the sequence.
    moving_image : ndarray
        The second gray scale image of the sequence.
    weights : ndarray
        Weights as an array with the same shape as reference_image
    num_warp : int
        Number of inner iteration
    tol : float
        Tolerance
    matrix : ndarray
        Initial homogeneous transformation matrix
        see https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html)

    Returns
    -------
    matrix : ndarray
        The estimated transformation matrix

    Note
    ----
    This is an inner function for parametric_ilk

    """

    ndim = reference_image.ndim

    grid = np.meshgrid(*[np.arange(n) for n in reference_image.shape], indexing="ij")

    grad = np.gradient(reference_image)

    elems = [*grad, *[x * dx for dx, x in product(grad, grid)]]

    G = np.zeros([len(elems)] * 2)

    for i, j in combinations_with_replacement(range(len(elems)), 2):
        G[i, j] = G[j, i] = (elems[i] * elems[j] * weights).sum()

    Id = np.eye(ndim, dtype=matrix.dtype)

    for _ in range(num_warp):
        moving_image_warp = ndimage.affine_transform(moving_image, matrix)
        error_image = reference_image - moving_image_warp
        b = np.array([[(e * error_image * weights).sum()] for e in elems])
        try:
            r = np.linalg.solve(G, b)
            matrix[:ndim, -1] += (matrix[:ndim, :ndim] @ r[:ndim]).ravel()
            matrix[:ndim, :ndim] @= r[ndim:].reshape(ndim, ndim) + Id
            if np.linalg.norm(r) < tol:
                break
        except np.linalg.LinAlgError:
            break

    return matrix


def _huber_weights(x, k):
    """Huber weights

    Parameters
    ----------
    x : ndarray
        input
    k : float
        scale parameter

    Result
    ------
    w : array
        the weights for iterative least square
    """
    ax = np.abs(x)
    w = np.ones(x.shape)
    w[ax > k] = k / x[ax > k]
    return w


def _parametric_rilk_solver(
    reference_image, moving_image, weights, num_warp, tol, matrix
):
    """Estimate affine motion between two images using iterative Lucas Kanade approach

    Parameters
    ----------

    reference_image : ndarray
        The first gray scale image of the sequence.
    moving_image : ndarray
        The second gray scale image of the sequence.
    weights : ndarray
        Weights as an array with the same shape as reference_image
    num_warp : int
        Number of inner iteration
    tol : float
        Tolerance
    matrix: np.ndarray
        Transformation matrix

    Returns
    -------
    matrix : ndarray
        Transformation matrix

    """

    ndim = reference_image.ndim

    grid = np.meshgrid(*[np.arange(n) for n in reference_image.shape], indexing="ij")

    grad = [ndimage.prewitt(reference_image, axis=k) for k in range(ndim)]

    elems = [*grad, *[x * dx for dx, x in product(grad, grid)]]

    G = np.zeros([len(elems)] * 2)

    Id = np.eye(ndim, dtype=matrix.dtype)

    for _ in range(num_warp):
        moving_image_warp = ndimage.affine_transform(moving_image, matrix)

        error_image = reference_image - moving_image_warp

        w = weights * _huber_weights(error_image, 4 * error_image.std())

        for i, j in combinations_with_replacement(range(len(elems)), 2):
            G[i, j] = G[j, i] = (elems[i] * elems[j] * w).sum()

        b = np.array([[(e * error_image * w).sum()] for e in elems])

        try:
            r = np.linalg.solve(G, b)
            matrix[:ndim, -1] += (matrix[:ndim, :ndim] @ r[:ndim]).ravel()
            matrix[:ndim, :ndim] @= r[ndim:].reshape(ndim, ndim) + Id
            if np.linalg.norm(r) < tol:
                break
        except np.linalg.LinAlgError:
            break

    return matrix


def parametric_ilk(
    reference_image,
    moving_image,
    *,
    weights=None,
    num_warp=100,
    tol=1e-3,
    pyramid_downscale=2,
    pyramid_minimum_size=16,
    matrix=None,
    dtype=np.float32,
):
    """Estimate affine motion between two images using an
    iterative Lucas and Kanade approach

    Parameters
    ----------
    reference_image : ndarray
        The first gray scale image of the sequence.
    moving_image : ndarray
        The second gray scale image of the sequence.
    weights : ndarray or None
        The weights array with same shape as reference_image
    num_warp : int
        Maximumnumber of inner iteration
    tol : float
        Tolerance for inner iterations
    pyramid_scale : float
        The pyramid downscale factor.
    pyramid_minimum_size : int
        The minimum size for any dimension of the pyramid levels.
    matrix : ndarray
        Intial guess for the homogeneous transformation matrix
    dtype : dtype
        Output data type.

    Returns
    -------
    matrix: ndarray
        Transformation matrix

    Reference
    ---------
    .. [1] http://robots.stanford.edu/cs223b04/algo_affine_tracking.pdf

    Example
    -------

    >>> from skimage import data
    >>> import numpy as np
    >>> from scipy import ndimage as ndi
    >>> from skimage.registration import parametric_ilk
    >>> import matplotlib.pyplot as plt
    >>> reference_image = data.astronaut()[...,0]
    >>> r = -0.12  # radians
    >>> c, s = np.cos(r), np.sin(r)
    >>> A0 =  np.array([[c, -s], [s, c]])
    >>> v0 = np.random.randn(2).flatten()
    >>> moving_image = ndi.affine_transform(reference_image, A0, v0)
    >>> A1,v1 = parametric_ilk(reference_image, moving_image)
    >>> registered_moving = ndi.affine_transform(moving_image, A1, v1.ravel())
    >>> plt.imshow(registered_moving)

    """

    ndim = reference_image.ndim

    if matrix is None:
        matrix = np.eye(ndim + 1, dtype=np.float64)

    solver = partial(_parametric_ilk_solver, num_warp=num_warp, tol=tol)

    if weights is None:
        weights = np.ones(reference_image.shape, dtype=dtype)

    return _coarse_to_fine_parametric(
        reference_image,
        moving_image,
        weights,
        solver,
        pyramid_downscale,
        pyramid_minimum_size,
        matrix,
        dtype,
    )


def _cost_nmi(x, reference_image, moving_image):
    """Negative normalized mutual information

    Parameters
    ----------
    x: ndarray
        Parameters of the transform
    reference_image: ndarray
        Reference image
    moving_image: ndarray
        Moving image
    """
    ndim = reference_image.ndim
    matrix = np.eye(ndim + 1, dtype=np.float64)
    matrix[:ndim, :] = x.reshape(ndim, ndim + 1)
    moving_image_warp = ndimage.affine_transform(moving_image, matrix)
    return -normalized_mutual_information(reference_image, moving_image_warp)


def _parametric_nmi_solver(reference_image, moving_image, weights, tol, matrix):
    """Solver maximizing mutual information

    Parameters
    ----------
    reference_image : ndarray
        The first gray scale image of the sequence.
    moving_image : ndarray
        The second gray scale image of the sequence.
    weights: ndarray
        Weights
    tol: float
        Solver tolerance
    matrix: ndarray
        Initial value of the transform

    Note:
    from PR #3544
    """

    ndim = reference_image.ndim

    cost = partial(
        _cost_nmi, reference_image=reference_image, moving_image=moving_image
    )

    x0 = matrix[:ndim, :].ravel()

    result = minimize(cost, x0=x0, method="Powell", tol=tol)
    matrix = np.eye(ndim + 1, dtype=np.float64)
    matrix[:ndim, :] = result.x.reshape(ndim, ndim + 1)
    return matrix


def parametric_nmi(
    reference_image,
    moving_image,
    *,
    weights=None,
    tol=None,
    pyramid_scale=2,
    pyramid_minimum_size=32,
    matrix=None,
    dtype=np.float32,
):
    """Estimate affine motion between two images by maximizing mutual
    information

    Parameters
    ----------
    reference_image : ndarray
        The first gray scale image of the sequence.
    moving_image : ndarray
        The second gray scale image of the sequence.
    weights : ndarray or None
        The weights array with same shape as reference_image
    tol : float
        Tolearance for the solver (see scipy.mininize).
    downscale : float
        The pyramid downscale factor.
    nlevel : int
        The maximum number of pyramid levels.
    min_size : int
        The minimum size for any dimension of the pyramid levels.
    matrix : ndarray
        Intial guess for the homogeneous transformation matrix
    dtype : dtype
        Output data type.

    Returns
    -------
    matrix: ndarray
        Transformation matrix

    Reference
    ---------
    .. [1] Studholme C, Hill DL, Hawkes DJ. Automated 3-D registration of MR
        and CT images of the head. Med Image Anal. 1996 Jun;1(2):163-75.

    Example
    -------

    >>> from skimage import data
    >>> import numpy as np
    >>> from scipy import ndimage as ndi
    >>> from skimage.registration import parametric_ilk
    >>> import matplotlib.pyplot as plt
    >>> reference_image = data.astronaut()[...,0]
    >>> r = -0.12  # radians
    >>> c, s = np.cos(r), np.sin(r)
    >>> A0 =  np.array([[c, -s], [s, c]])
    >>> v0 = np.random.randn(2).flatten()
    >>> moving_image = ndi.affine_transform(reference_image, A0, v0)
    >>> A1,v1 = parametric_ilk(reference_image, moving_image)
    >>> registered_moving = ndi.affine_transform(moving_image, A1, v1.ravel())
    >>> plt.imshow(registered_moving)

    """

    ndim = reference_image.ndim

    if matrix is None:
        matrix = np.eye(ndim + 1, dtype=np.float64)

    solver = partial(_parametric_nmi_solver, tol=tol)

    if weights is None:
        weights = np.ones(reference_image.shape, dtype=dtype)

    return _coarse_to_fine_parametric(
        reference_image,
        moving_image,
        weights,
        solver,
        pyramid_scale,
        pyramid_minimum_size,
        matrix,
        dtype,
    )
