from itertools import product, combinations_with_replacement
from functools import partial
import numpy as np
from scipy import ndimage
from skimage.registration._optical_flow_utils import _get_pyramid

"""Parametric registration"""

"""
TODO: have a single object for the transform
TODO: handle other parametric motion: translation, rotation, etc
TODO: handle color images? channel_axis?
TODO: add other solvers
TODO: merge with PR #3544 https://github.com/seanbudd/scikit-image and #7050 https://github.com/Coilm/scikit-image
TODO: add more test
"""


def _coarse_to_fine_parametric(
    I0, I1, weights, solver, downscale=2, nlevel=10, min_size=16, dtype=np.float32
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
    downscale : float
        The pyramid downscale factor.
    nlevel : int
        The maximum number of pyramid levels.
    min_size : int
        The minimum size for any dimension of the pyramid levels.
    dtype : dtype
        Output data type.

    Returns
    -------
    A : ndarray
        The estimated affine matrix
    v : ndarray
        The estimated offset

    Note
    ----
    This function follows skimage.registration._optical_flow_utils.coarse_to_fine

    Generate image pyramids and apply the solver at each level passing the
    previous affine parameters from the previous estimate.

    """

    ndim = I0.ndim

    pyramid = list(
        zip(
            _get_pyramid(I0.astype(dtype), downscale, nlevel, min_size),
            _get_pyramid(I1.astype(dtype), downscale, nlevel, min_size),
            _get_pyramid(weights.astype(dtype), downscale, nlevel, min_size),
        )
    )

    A = np.eye(ndim)
    v = np.zeros((ndim, 1))

    A, v = solver(pyramid[0][0], pyramid[0][1], pyramid[0][2], A=A, v=v)
    for J0, J1, W in pyramid[1:]:
        A, v = solver(J0, J1, W, A=A, v=downscale * v)

    return A, v


def _parametric_ilk_solver(reference_image, moving_image, weights, num_warp, A, v):
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
    A : ndarray
        Affine matrix shape (ndim, ndim)
    v : ndarray
        Offset (ndim,)

    Returns
    -------
    A : ndarray
        The estimated affine matrix
    v : ndarray
        The estimated offset

    Note
    ----
    This is an inner function for parametric_ilk

    """

    ndim = reference_image.ndim

    grid = np.meshgrid(*[np.arange(n) for n in reference_image.shape], indexing="ij")

    grad = [ndimage.prewitt(reference_image, axis=k) for k in range(ndim)]

    elems = [*grad, *[x * dx for dx, x in product(grad, grid)]]

    G = np.zeros([len(elems)] * 2)
    for i, j in combinations_with_replacement(range(len(elems)), 2):
        G[i, j] = G[j, i] = (elems[i] * elems[j] * weights).sum()

    for _ in range(num_warp):
        moving_image_warp = ndimage.affine_transform(moving_image, A, v.flatten())
        error_image = reference_image - moving_image_warp
        b = np.array([[(e * error_image * weights).sum()] for e in elems])
        try:
            r = np.linalg.solve(G, b)
            v = v + A @ r[:ndim]
            A = A @ (r[ndim:].reshape(ndim, ndim) + np.eye(ndim))
            if np.linalg.norm(r) < 1e-9:
                break
        except np.linalg.LinAlgError:
            break

    return A, v


def _huber_weights(x, k):
    """Huber weights

    Parameters
    ----------
    x : array
        input
    k : scalar
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


def _parametric_rilk_solver(reference_image, moving_image, weights, num_warp, A, v):
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
    A : ndarray
        Affine matrix shape (ndim, ndim)
    v : ndarray
        Offset (ndim,)

    Returns
    -------
    A : ndarray
        The estimated affine matrix
    v : ndarray
        The estimated offset

    Note
    ----
    This is an inner function for  parametric_ilk

    """

    ndim = reference_image.ndim

    grid = np.meshgrid(*[np.arange(n) for n in reference_image.shape], indexing="ij")

    grad = [ndimage.prewitt(reference_image, axis=k) for k in range(ndim)]

    elems = [*grad, *[x * dx for dx, x in product(grad, grid)]]

    G = np.zeros([len(elems)] * 2)

    for _ in range(num_warp):
        moving_image_warp = ndimage.affine_transform(moving_image, A, v.flatten())

        error_image = reference_image - moving_image_warp

        w = weights * _huber_weights(error_image, 4 * error_image.std())

        for i, j in combinations_with_replacement(range(len(elems)), 2):
            G[i, j] = G[j, i] = (elems[i] * elems[j] * w).sum()

        b = np.array([[(e * error_image * w).sum()] for e in elems])

        try:
            r = np.linalg.solve(G, b)
            v = v + A @ r[:ndim]
            A = A @ (r[ndim:].reshape(ndim, ndim) + np.eye(ndim))
            if np.linalg.norm(r) < 1e-5:
                break
        except np.linalg.LinAlgError:
            break

    return A, v


def parametric_ilk(
    reference_image,
    moving_image,
    *,
    weights=None,
    robust=False,
    num_warp=100,
    downscale=2,
    nlevel=20,
    min_size=16,
    A=None,
    v=None,
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
        Number of inner iteration
    downscale : float
        The pyramid downscale factor.
    nlevel : int
        The maximum number of pyramid levels.
    min_size : int
        The minimum size for any dimension of the pyramid levels.
    dtype : dtype
        Output data type.

    Returns
    -------
    A : ndarray
        The estimated affine matrix
    v : ndarray
        The estimated offset

    Reference
    ---------
    .. [1] http://robots.stanford.edu/cs223b04/algo_affine_tracking.pdf

    Example
    -------

    from skimage import data
    import numpy as np
    from scipy import ndimage as ndi
    from skimage.registration import parametric_ilk
    import matplotlib.pyplot as plt
    reference_image = data.astronaut()[...,0]
    r = -0.12  # radians
    c, s = np.cos(r), np.sin(r)
    A0 =  np.array([[c, -s], [s, c]])
    v0 = np.random.randn(2).flatten()
    moving_image = ndi.affine_transform(reference_image, A0, v0)
    A1,v1 = parametric_ilk(reference_image, moving_image)
    registered_moving = ndi.affine_transform(moving_image, A1, v1.ravel())
    plt.imshow(registered_moving)

    """
    ndim = reference_image.ndim

    if A is None:
        A = np.eye(ndim)

    if v is None:
        v = np.zeros((ndim,))

    if robust:
        solver = partial(_parametric_rilk_solver, num_warp=num_warp, A=A, v=v)
    else:
        solver = partial(_parametric_ilk_solver, num_warp=num_warp, A=A, v=v)

    if weights is None:
        weights = np.ones(reference_image.shape, dtype=dtype)

    return _coarse_to_fine_parametric(
        reference_image,
        moving_image,
        weights,
        solver,
        downscale,
        nlevel,
        min_size,
        dtype,
    )
