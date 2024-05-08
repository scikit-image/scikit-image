from itertools import product, combinations_with_replacement
from functools import partial
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize
from skimage.metrics import normalized_mutual_information
from skimage.transform import pyramid_gaussian
from math import log, floor

"""Affine image registration

TODO: handle other parametric motion: translation, rotation, etc
TODO: handle color images? channel_axis?
TODO: merge with PR #3544 https://github.com/seanbudd/scikit-image and #7050 https://github.com/Coilm/scikit-image
TODO: add more test (weights, color)
TODO: name of the functions
"""


def lucas_kanade_affine_solver(
    reference_image,
    moving_image,
    weights,
    channel_axis,
    matrix,
    *,
    max_iter=40,
    tol=1e-6,
):
    """Estimate affine motion between two images using a least square approach

    Parameters
    ----------

    reference_image : ndarray
        The first gray scale image of the sequence.
    moving_image : ndarray
        The second gray scale image of the sequence.
    weights : ndarray
        Weights as an array with the same shape as reference_image
    channel_axis: int
        Index of the channel axis
    matrix : ndarray
        Initial homogeneous transformation matrix
    max_iter : int
        Number of inner iteration
    tol : float
        Tolerance

    Returns
    -------
    matrix : ndarray
        The estimated transformation matrix

    Reference
    ---------
    .. [1] http://robots.stanford.edu/cs223b04/algo_affine_tracking.pdf

    """

    if weights is None:
        weights = 1.0

    if channel_axis is not None:
        reference_image = np.moveaxis(reference_image, channel_axis, 0)
        moving_image = np.moveaxis(moving_image, channel_axis, 0)
    else:
        reference_image = np.expand_dims(reference_image, 0)
        moving_image = np.expand_dims(moving_image, 0)

    ndim = reference_image.ndim - 1

    grid = np.meshgrid(
        *[np.arange(n) for n in reference_image.shape[1:]], indexing="ij"
    )

    grad = [
        np.gradient(reference_image, axis=k)[0] for k in range(1, reference_image.ndim)
    ]

    elems = [*grad, *[x * dx for dx, x in product(grad, grid)]]

    G = np.zeros([len(elems)] * 2)

    for i, j in combinations_with_replacement(range(len(elems)), 2):
        G[i, j] = G[j, i] = (elems[i] * elems[j] * weights).mean()

    Id = np.eye(ndim, dtype=matrix.dtype)

    for _ in range(max_iter):
        moving_image_warp = np.stack(
            [ndi.affine_transform(plane, matrix) for plane in moving_image]
        )
        error_image = reference_image - moving_image_warp
        b = np.array([[(e * error_image * weights).mean()] for e in elems])
        try:
            r = np.linalg.solve(G, b)
            matrix[:ndim, -1] += (matrix[:ndim, :ndim] @ r[:ndim]).ravel()
            matrix[:ndim, :ndim] @= r[ndim:].reshape(ndim, ndim) + Id
            if np.linalg.norm(r) < tol:
                break
        except np.linalg.LinAlgError:
            break

    return matrix


def _cost_nmi(x, reference_image, moving_image, weights):
    """Negative normalized mutual information

    Parameters
    ----------
    x: ndarray
        Parameters of the transform
    reference_image: ndarray
        Reference image
    moving_image: ndarray
        Moving image
    weights: ndarray | None
        Weights
    """

    ndim = reference_image.ndim - 1

    matrix = np.eye(ndim + 1, dtype=np.float64)
    matrix[:ndim, :] = x.reshape(ndim, ndim + 1)

    moving_image_warp = np.stack(
        [ndi.affine_transform(plane, matrix) for plane in moving_image]
    )

    return -normalized_mutual_information(
        reference_image, moving_image_warp, weights=weights
    )


def studholme_affine_solver(
    reference_image,
    moving_image,
    weights,
    channel_axis,
    matrix,
    *,
    options={"maxiter": 10},
):
    """Solver maximizing mutual information using Powell's method

    Parameters
    ----------
    reference_image : ndarray
        The first gray scale image of the sequence.
    moving_image : ndarray
        The second gray scale image of the sequence.
    weights: ndarray
        Weights or mask
    channel_axis: int | None
        Index of the channel axis
    matrix: ndarray
        Initial value of the transform
    options: dict
        options for scipy.optimization.minimize("Powell")

    Returns
    -------
    matrix:
        Homogeneous transform matrix

    Reference
    ---------
    .. [1] Studholme C, Hill DL, Hawkes DJ. Automated 3-D registration of MR
        and CT images of the head. Med Image Anal. 1996 Jun;1(2):163-75.
    .. [2] J. Nunez-Iglesias, S. van der Walt, and H. Dashnow, Elegant SciPy:
        The Art of Scientific Python. O’Reilly Media, Inc., 2017.

    Note:
    from PR #3544
    """

    if channel_axis is not None:
        reference_image = np.moveaxis(reference_image, channel_axis, 0)
        moving_image = np.moveaxis(moving_image, channel_axis, 0)
    else:
        reference_image = np.expand_dims(reference_image, 0)
        moving_image = np.expand_dims(moving_image, 0)

    ndim = reference_image.ndim - 1

    cost = partial(
        _cost_nmi,
        reference_image=reference_image,
        moving_image=moving_image,
        weights=weights,
    )

    x0 = matrix[:ndim, :].ravel()

    result = minimize(cost, x0=x0, method="Powell", options=options)

    matrix = np.eye(ndim + 1, dtype=np.float64)
    matrix[:ndim, :] = result.x.reshape(ndim, ndim + 1)
    return matrix


def affine(
    reference_image,
    moving_image,
    *,
    weights=None,
    channel_axis=None,
    matrix=None,
    solver=lucas_kanade_affine_solver,
    pyramid_downscale=2,
    pyramid_minimum_size=32,
):
    """Coarse-to-fine affine motion estimation between two images

    Parameters
    ----------
    reference_image : ndarray
        The first gray scale image of the sequence.
    moving_image : ndarray
        The second gray scale image of the sequence.
    weights : ndarray | None
        The weights array with same shape as reference_image
    channel_axis: int | None
        Index of the channel axis
    matrix : ndarray
        Intial guess for the homogeneous transformation matrix
    solver: lambda(reference_image, moving_image, weights, channel_axis, matrix) -> matrix
        Affine motion solver can be lucas_kanade_affine_solver or studholme_affine_solver
    pyramid_scale : float
        The pyramid downscale factor.
    pyramid_minimum_size : int
        The minimum size for any dimension of the pyramid levels.

    Returns
    -------
    matrix: ndarray
        Transformation matrix

    Note
    ----
    Generate image pyramids and apply the solver at each level passing the
    previous affine parameters from the previous estimate.

    The estimated matrix can be used with scikit.ndimage.affine to register the moving image
    to the reference image.

    Example
    -------
    >>> from skimage import data
    >>> import numpy as np
    >>> from scipy import ndimage as ndi
    >>> from skimage.registration import affine
    >>> import matplotlib.pyplot as plt
    >>> reference = data.astronaut()[...,0]
    >>> r = -0.12  # radians
    >>> c, s = np.cos(r), np.sin(r)
    >>> matrix_transform = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    >>> moving_image = ndi.affine_transform(reference_image, matrix_transform)
    >>> matrix = affine(reference_image, moving_image)
    >>> registered_moving = ndi.affine_transform(moving_image, matrix)
    >>> plt.imshow(registered_moving)

    """

    ndim = reference_image.ndim if channel_axis is None else reference_image.ndim - 1

    if matrix is None:
        matrix = np.eye(ndim + 1, dtype=np.float64)

    if channel_axis is not None:
        shape = [d for k, d in enumerate(reference_image.shape) if k != channel_axis]
    else:
        shape = reference_image.shape

    max_layer = int(
        floor(
            log(min(shape), pyramid_downscale)
            - log(pyramid_minimum_size, pyramid_downscale)
        )
    )

    pyramid = list(
        zip(
            *[
                reversed(
                    list(
                        pyramid_gaussian(
                            x,
                            max_layer=max_layer,
                            downscale=pyramid_downscale,
                            preserve_range=True,
                            channel_axis=channel_axis,
                        )
                        if x is not None
                        else [None] * (max_layer + 1)
                    )
                )
                for x in [reference_image, moving_image, weights]
            ]
        )
    )

    matrix = solver(
        pyramid[0][0],
        pyramid[0][1],
        pyramid[0][2],
        channel_axis=channel_axis,
        matrix=matrix,
    )

    for J0, J1, W in pyramid[1:]:
        scaled_matrix = matrix
        scaled_matrix[:ndim, -1] = pyramid_downscale * scaled_matrix[:ndim, -1]
        matrix = solver(J0, J1, W, channel_axis=channel_axis, matrix=scaled_matrix)

    return matrix
