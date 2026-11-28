"""Images that are sums of Gaussians, with exact smoothed derivatives.

Tests use these images as gold standards for Gaussian filters and their
derivatives. `fit` approximates a photograph by a Mobahi Gaussian RBF sum [1];
`render` and `hessian` then give the sum, smoothed by a Gaussian of any scale,
and its derivatives, in closed form. The ``gaussian_sum_images`` fixture in
``tests/conftest.py`` holds such sums for three photographs.

Coordinates are in pixels, in skimage ``rc`` order: ``r`` down the rows, ``c``
across the columns, with pixel centres at integers.

Smoothing identity (Mobahi 2016, §2.2) [1]::

    f(x) = Σ_j a_j exp(−‖x − x_j‖² / (2 δ_j²))
    [f ⋆ k_σ](x) = Σ_j a_j (δ_j / √(δ_j² + σ²))^n
                     · exp(−‖x − x_j‖² / (2 (δ_j² + σ²))),   n = 2

j indexes Gaussians; each Gaussian has a width δ_j, a centre x_j, and an
amplitude a_j. We choose the δ_j and x_j, then do a least-squares fit to find
the a_j.

In our chosen scheme, we place Gaussians on a regular grid. Centres of one
width δ form a *level*. On one level we have a grid of Gaussians with spacing
δ, reaching one spacing past each edge. We then fit the a_j level by level,
from coarse (large δ) to fine (small δ), each level to the residual that the
coarser levels leave.

The order of the fit has little effect. A joint fit puts the row bases of all
levels side by side, then the column bases, and does the two solves below
once. Its amplitudes then include cross terms between widths, such as a row
Gaussian of width 16 times a column Gaussian of width 1.5, which are not
isotropic. With the ridge set to give the same residual, the joint fit is
about as good as this one, but it has more than three times as many
amplitudes. Repeating the coarse-to-fine pass, with each level refitted to
the residual of all the others, does not change the residual. Coarse to fine
keeps each level small and every term isotropic. See the workbook [3], §3.

On our grid, Mobahi's single index j becomes a pair (j, k): row centre r_j
and column centre c_k.

The 2-D basis separates into row and column factors because an isotropic
Gaussian factors along axes, as the squared Euclidean norm does::

    exp(−((r−r_j)² + (c−c_k)²) / (2 δ²))
      = exp(−(r−r_j)² / (2 δ²)) · exp(−(c−c_k)² / (2 δ²)).

The first factor depends only on the row and row-centre index j; the second
only on the column and column-centre index k. Factoring alone is not enough
for a cheap fit: an unstructured cloud of centres would still give product
kernels, but not a *product set* of kernels — the design would be one
independent 2-D bump per centre, with no Kronecker structure.

A rectangular grid supplies that structure. Every centre is a pair (r_j, c_k)
from fixed 1-D lists, and every pair appears once, so the 2-D dictionary is
the outer product of two 1-D dictionaries. Columns of A_r are the row
Gaussians on the image rows; columns of A_c are the column Gaussians on the
image columns. One amplitude a_jk per grid point is Mobahi's a_j for that
centre. The model for the residual Y at one width is::

    Y ≈ A_r @ A @ A_c.T
      = Σ_{j,k} a_jk · (row Gaussian_j) · (column Gaussian_k)^T,

with A = (a_jk) a full matrix — not forced to rank one. Separability lives in
the basis, not in the coefficients.

Vectorising gives (A_c ⊗ A_r) vec(A) ≈ vec(Y), a Kronecker least-squares
problem (Fausett & Fulton 1994) [2]. Rather than form the Kronecker product,
two ridge least-squares solves fit the separable model: A_r M ≈ Y, then
A_c A.T ≈ M.T.

We use ridge least squares because neighbouring Gaussians, spaced δ apart
with width δ, overlap heavily. Without a penalty, the fit uses large
amplitudes that nearly cancel. The ridge keeps the amplitudes small, at a
small cost in fit accuracy, and lowers the sigma-1 Hessian error of the
filter.

References
----------
.. [1] H. Mobahi, "Closed Form for Some Gaussian Convolutions,"
    arXiv:1602.05610 [math.CA], 2016.
.. [2] D. W. Fausett and C. T. Fulton, "Large Least Squares Problems Involving
    Kronecker Products," SIAM J. Matrix Anal. Appl., 15(1):219–227, 1994.
    https://doi.org/10.1137/S0895479891222106
.. [3] https://scikit-image.org/skimage-workbooks/on-gaussian-sum-images
"""

from dataclasses import dataclass

import numpy as np
import numpy.linalg as npl

DELTAS = (16.0, 8.0, 4.0, 2.0, 1.5)  # widths δ, pixels, coarse to fine
RIDGE = 1e-3  # relative ridge penalty for the separable least-squares fit
NDIM = 2  # n = dim(x) for images


@dataclass
class Level:
    """Gaussians of one width δ, centred on a grid of points (r_j, c_k)."""

    delta: float
    row_centres: np.ndarray
    col_centres: np.ndarray
    amplitudes: np.ndarray  # a_jk; shape (len(row_centres), len(col_centres))


def grid_centres(n, spacing):
    """Centres along an axis of length ``n``, one spacing past each edge."""
    count = int(np.ceil((n - 1 + 2 * spacing) / spacing)) + 1
    return -spacing + spacing * np.arange(count)


def basis(coords, centres, delta, order=0):
    """1-D factors ``exp(−(t − t_j)² / (2 δ²))``, or a derivative."""
    offset = coords[:, None] - centres[None, :]
    gaussian = np.exp(-(offset**2) / (2 * delta**2))
    if order == 0:
        return gaussian
    if order == 1:
        return -offset / delta**2 * gaussian
    if order == 2:
        return (offset**2 - delta**2) / delta**4 * gaussian
    raise ValueError(f'order must be 0, 1 or 2, not {order}')


def ridge_solve(X, Y, ridge):
    """Ridge regression of ``Y`` on ``X``, as least squares on augmented data.

    In the notation of Wikipedia's "Ridge regression", with design matrix
    ``X``, response ``Y`` (here one column per right-hand side) and
    coefficients ``β``, the model is ``Y = X β + ε``, and the ridge estimate
    minimizes

        ‖Y − X β‖² + λ ‖β‖²,

    with solution

        β̂_λ = (Xᵀ X + λ I)⁻¹ Xᵀ Y.

    This is ordinary least squares on the augmented data

        X_aug = [X; √λ I],    Y_aug = [Y; 0],

    because ``X_augᵀ X_aug = Xᵀ X + λ I`` and ``X_augᵀ Y_aug = Xᵀ Y``.

    The penalty is ``λ = ridge × s₀²``, where ``s₀`` is the largest singular
    value of ``X``. ``X_aug`` has singular values ``√(s_i² + λ)``, so it is
    well conditioned.

    https://en.wikipedia.org/wiki/Ridge_regression
    """
    lambda_ = ridge * npl.norm(X, 2) ** 2
    n_coef, n_cols = X.shape[1], Y.shape[1]
    X_aug = np.vstack([X, np.sqrt(lambda_) * np.eye(n_coef)])
    Y_aug = np.vstack([Y, np.zeros((n_coef, n_cols))])
    return npl.lstsq(X_aug, Y_aug)[0]


def fit(image, deltas=DELTAS, ridge=RIDGE):
    """Fit levels coarse to fine. Return the levels and the residual.

    The Gaussians of one width ``δ`` sit on a grid of spacing ``δ``, reaching
    one spacing past each edge. Each level is fitted to the residual that the
    coarser levels leave.

    An isotropic Gaussian factors into a row Gaussian times a column Gaussian.
    On a rectangular grid of centres, the 2-D basis is therefore the outer
    product of two 1-D bases: ``A_r`` (row Gaussians on the image rows) and
    ``A_c`` (column Gaussians on the image columns). The model at one width is

        residual ≈ A_r @ A @ A_c.T,

    with ``A`` the full matrix of amplitudes ``a_jk``. Vectorised, this is
    ``(A_c ⊗ A_r) vec(A) ≈ vec(residual)`` [2]. Two 1-D ridge solves fit it
    without forming the Kronecker product: ``A_r M ≈ residual``, then
    ``A_c A.T ≈ M.T``. That is a two-step ridge, not one ridge solve on the
    vectorised model.

    ``ridge`` is a relative Tikhonov penalty (see `ridge_solve`). Raising it
    shrinks the amplitudes and the cancellation between them, at the cost of
    fit accuracy.
    """
    residual = np.array(image, dtype=float)
    rows = np.arange(residual.shape[0], dtype=float)
    cols = np.arange(residual.shape[1], dtype=float)
    levels = []
    for delta in deltas:
        row_centres = grid_centres(len(rows), delta)
        col_centres = grid_centres(len(cols), delta)
        row_basis = basis(rows, row_centres, delta)
        col_basis = basis(cols, col_centres, delta)
        row_fit = ridge_solve(row_basis, residual, ridge)
        amplitudes = ridge_solve(col_basis, row_fit.T, ridge).T
        residual = residual - row_basis @ amplitudes @ col_basis.T
        levels.append(Level(delta, row_centres, col_centres, amplitudes))
    return levels, residual


def render(levels, shape, sigma=0.0, order=(0, 0)):
    """Mobahi's smoothed sum, or a derivative of that sum.

    The levels define Mobahi's Gaussian RBF sum ([1], §2.2, unnumbered),

        f(x) = Σ_j a_j exp(−‖x − x_j‖² / (2 δ_j²)).

    Its convolution with the normalised Gaussian ``k_σ`` ([1], §1,
    "Notation") is closed form ([1], §2.2, unnumbered; stated and proved as
    Lemma 1 in Appendix B):

        [f ⋆ k_σ](x) = Σ_j a_j (δ_j / √(δ_j² + σ²))^n
                           exp(−‖x − x_j‖² / (2 (δ_j² + σ²))).

    So each level of width ``δ`` and amplitudes ``a_j`` becomes width
    ``√(δ² + σ²)`` with amplitudes ``a_j (δ / √(δ² + σ²))^n``. At ``σ = 0``
    this is ``f`` itself.

    The derivatives are not in [1]. Each term factors into a row Gaussian
    times a column Gaussian, so the ``(p, q)`` derivative differentiates the
    row factor ``p`` times and the column factor ``q`` times (`basis`).
    ``order`` is ``(row derivative, column derivative)``. Values are at pixel
    centres.
    """
    rows = np.arange(shape[0], dtype=float)
    cols = np.arange(shape[1], dtype=float)
    out = np.zeros((len(rows), len(cols)))
    for level in levels:
        # [1], Lemma 1: width δ becomes √(δ² + σ²) ...
        delta_s = np.sqrt(level.delta**2 + sigma**2)
        # ... and amplitudes a_j are multiplied by (δ / √(δ² + σ²))^n.
        scale = (level.delta / delta_s) ** NDIM
        # One separable term per centre: (row factor) a_jk (column factor).
        row_basis = basis(rows, level.row_centres, delta_s, order[0])
        col_basis = basis(cols, level.col_centres, delta_s, order[1])
        out += scale * row_basis @ level.amplitudes @ col_basis.T
    return out


def hessian(levels, shape, sigma):
    """``[Hrr, Hrc, Hcc]`` of Mobahi's smoothed sum at scale ``sigma``."""
    return [render(levels, shape, sigma, order) for order in ((2, 0), (1, 1), (0, 2))]
