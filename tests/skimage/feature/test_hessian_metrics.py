"""Accuracy of ``hessian_matrix`` against closed-form Hessians.

``hessian_matrix(..., use_gaussian_derivatives=True)`` filters each axis once,
with a sampled Gaussian derivative whose discrete moments have been restored.
These tests measure how close that discrete filter stays to two continuous
standards.

1. Mobahi's Gaussian RBF sums [1], fitted to photographs (``camera``,
   ``coins``, greyscale ``astronaut``). Gaussian smoothing maps such a sum to
   another sum of the same form, so its Hessian is exact at every scale
   (`render`). The amplitudes come from separable ridge least squares [2]
   (`fit`); the photograph only chooses them. Two measures compare the filter
   with this standard: the largest error of any element, relative to the peak
   (`max_relative_error`), and the relative RMS error of the whole matrix,
   split into a common gain and the rest (`gain_and_shape_error`).

2. Lindeberg's scale-space sinusoid [3]. Smoothing at scale ``t = σ²`` damps a
   wave of frequency ``ω`` by ``exp(−|ω|² t / 2)``, so the smoothed Hessian of a
   plane wave is ``−ω ωᵀ exp(−|ω|² t / 2)`` times the wave. One impulse gives
   the filter's response at every frequency (`frequency_error`). The
   photographs carry little energy near the Nyquist frequency, so this is the
   test that exposes aliasing there.

See https://scikit-image.org/skimage-workbooks/on-gaussian-sum-images.

References
----------
.. [1] H. Mobahi, "Closed Form for Some Gaussian Convolutions,"
    arXiv:1602.05610 [math.CA], 2016.
.. [2] D. W. Fausett and C. T. Fulton, "Large Least Squares Problems Involving
    Kronecker Products," SIAM J. Matrix Anal. Appl., 15(1):219–227, 1994.
    https://doi.org/10.1137/S0895479891222106
.. [3] T. Lindeberg, "Feature Detection with Automatic Scale Selection,"
    Int. J. Comput. Vis., 30(2):79–116, 1998, §3.
    https://doi.org/10.1023/A:1008045108935
"""

from dataclasses import dataclass

import numpy as np
import numpy.linalg as npl
import pytest

from skimage import data, img_as_float
from skimage.color import rgb2gray
from skimage.feature import hessian_matrix


def hessian(image, sigma, mode='nearest'):
    """The installed Gaussian-derivative Hessian, ``[Hrr, Hrc, Hcc]``."""
    return hessian_matrix(image, sigma=sigma, mode=mode, use_gaussian_derivatives=True)


# --- Mobahi Gaussian RBF sums fitted to photographs -----------------------
#
# Smoothing identity (Mobahi 2016, §2.2) [1]:
#
#   f(x) = Σ_j a_j exp(−‖x − x_j‖² / (2 δ_j²))
#   [f ⋆ k_σ](x) = Σ_j a_j (δ_j / √(δ_j² + σ²))^n
#                    · exp(−‖x − x_j‖² / (2 (δ_j² + σ²))),   n = 2
#
# Centres of one width δ form a level: a grid with spacing δ, reaching one
# spacing past each edge. Levels are fitted coarse to fine.
#
# The basis is separable because an isotropic Gaussian factors along axes; this
# is because the squared Euclidean norm does:
#
#   exp(−((r−r_j)² + (c−c_k)²) / (2 δ²))
#     = exp(−(r−r_j)² / (2 δ²)) · exp(−(c−c_k)² / (2 δ²)).
#
# The first factor depends only on the row and row-centre index j; the
# second only on the column and column-centre index k. Factoring alone is
# not enough for a cheap fit: an unstructured cloud of centres would still
# give product kernels, but not a *product set* of kernels — the design
# would be one independent 2-D bump per centre, with no Kronecker structure.
# A rectangular grid supplies that structure. Every centre is a pair
# (r_j, c_k) from fixed 1-D lists, and every pair appears once, so the 2-D
# dictionary is the outer product of two 1-D dictionaries. Columns of A_r
# are the row Gaussians on the image rows; columns of A_c are the column
# Gaussians on the image columns. One amplitude a_jk per grid point is
# Mobahi's a_j for that centre. The model at one width is
#
#   Y ≈ A_r @ A @ A_c.T
#     = Σ_{j,k} a_jk · (row Gaussian_j) · (column Gaussian_k)^T,
#
# with A = (a_jk) a full matrix — not forced to rank one. Separability lives
# in the basis, not in the coefficients.
#
# Vectorising gives (A_c ⊗ A_r) vec(A) ≈ vec(Y), a Kronecker least-squares
# problem (Fausett & Fulton 1994) [2]. Rather than form the Kronecker
# product, two ridge least-squares solves fit the separable model:
# A_r M ≈ Y, then A_c A.T ≈ M.T.

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


def _grid_centres(n, spacing):
    """Centres along an axis of length ``n``, one spacing past each edge."""
    count = int(np.ceil((n - 1 + 2 * spacing) / spacing)) + 1
    return -spacing + spacing * np.arange(count)


def _basis(coords, centres, delta, order=0):
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


def _ridge_solve(X, Y, ridge):
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

    ``ridge`` is a relative Tikhonov penalty (see `_ridge_solve`). Raising it
    shrinks the amplitudes and the cancellation between them, at the cost of
    fit accuracy.
    """
    residual = np.array(image, dtype=float)
    rows = np.arange(residual.shape[0], dtype=float)
    cols = np.arange(residual.shape[1], dtype=float)
    levels = []
    for delta in deltas:
        row_centres = _grid_centres(len(rows), delta)
        col_centres = _grid_centres(len(cols), delta)
        row_basis = _basis(rows, row_centres, delta)
        col_basis = _basis(cols, col_centres, delta)
        row_fit = _ridge_solve(row_basis, residual, ridge)
        amplitudes = _ridge_solve(col_basis, row_fit.T, ridge).T
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
    row factor ``p`` times and the column factor ``q`` times (`_basis`).
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
        row_basis = _basis(rows, level.row_centres, delta_s, order[0])
        col_basis = _basis(cols, level.col_centres, delta_s, order[1])
        out += scale * row_basis @ level.amplitudes @ col_basis.T
    return out


def exact_hessian(levels, shape, sigma):
    """``[Hrr, Hrc, Hcc]`` of Mobahi's smoothed sum at scale ``sigma``."""
    return [render(levels, shape, sigma, order) for order in ((2, 0), (1, 1), (0, 2))]


@pytest.fixture(scope='module')
def fitted_sums():
    """``{name: (levels, image)}``, ``image`` being the fitted sum at ``σ = 0``.

    The fits are checked to stay within 10% RMS of the photographs. A worse
    fit would not fail the Hessian tests, because their standard comes from
    the same fit, but it would make the test images less like photographs.
    """
    photographs = {
        'camera': img_as_float(data.camera()),
        'coins': img_as_float(data.coins()),
        'astronaut': img_as_float(rgb2gray(data.astronaut())),
    }
    fitted = {}
    for name, photograph in photographs.items():
        levels, residual = fit(photograph)
        relative_rms = np.sqrt(np.mean(residual**2) / np.mean(photograph**2))
        assert relative_rms < 0.10, (
            f'{name}: fit RMS residual is {relative_rms:.1%} of image RMS'
        )
        fitted[name] = (levels, render(levels, photograph.shape))
    return fitted


# --- Distance to a gold standard ------------------------------------------


def _interior(sigma):
    """Pixels an 8-sigma kernel reaches without crossing the frame.

    Outside these pixels the border rule supplies values that differ from the
    Gaussian sum's.
    """
    reach = int(8 * sigma + 0.5) + 1
    return (slice(reach, -reach),) * 2


def max_relative_error(got, truth, sigma):
    """Largest error of any element inside the frame, over the peak of ``truth``."""
    inner = _interior(sigma)
    peak = max(np.abs(t[inner]).max() for t in truth)
    return max(np.abs(g[inner] - t[inner]).max() for g, t in zip(got, truth)) / peak


FROBENIUS_WEIGHTS = (1, 2, 1)  # Hrr, Hrc, Hcc: Hrc is in the matrix twice


def gain_and_shape_error(got, truth, sigma):
    """Relative RMS error of the whole Hessian, split into gain and shape.

    With the Frobenius inner product summed over the pixels inside the frame,

        <A, B> = Σ_x (A_rr B_rr + 2 A_rc B_rc + A_cc B_cc),

    the relative RMS error is ``E = ‖got − truth‖ / ‖truth‖``. At one pixel
    ``‖H‖²`` is ``λ₁² + λ₂²``, so ``E`` does not change when the image axes
    rotate, and each pixel counts in proportion to its curvature.

    ``g = <got, truth> / ‖truth‖²`` is the one gain that best scales ``truth``
    onto ``got``. The residual ``got − g truth`` is orthogonal to ``truth``,
    so ``E² = (g − 1)² + S²`` with ``S = ‖got − g truth‖ / ‖truth‖``. A gain
    error scales every eigenvalue alike and leaves eigenvectors and eigenvalue
    ratios alone; ``S`` is the error that no one gain explains.

    Returns ``(g − 1, S)``.
    """
    inner = _interior(sigma)

    def frobenius(A, B):
        return sum(
            w * np.sum(a[inner] * b[inner]) for w, a, b in zip(FROBENIUS_WEIGHTS, A, B)
        )

    norm2 = frobenius(truth, truth)
    gain = frobenius(got, truth) / norm2
    rest = [g - gain * t for g, t in zip(got, truth)]
    return gain - 1, np.sqrt(frobenius(rest, rest) / norm2)


def frequency_error(sigma, band, size=257):
    """Largest error of the filter's frequency response for ``|ω| ≤ band × π``.

    The exact response of element ``(i, j)`` is
    ``−ω_i ω_j exp(−|ω|² σ² / 2)`` [3]. The measured response is the FFT of
    the filter's response to an impulse on the centre pixel of an odd frame,
    with zeros outside it (``mode='constant'``); 257 pixels hold an 8-sigma
    kernel at sigma 4. ``ifftshift`` moves the centre pixel to the origin that
    ``fft2`` expects. Each impulse response is symmetric about the centre, so
    its FFT is real to rounding.

    The error is divided by the peak of the exact ``Hrr`` response, so it is
    relative to the largest response, not to the response at each frequency.
    ``band`` is a fraction of the Nyquist frequency.
    """
    impulse = np.zeros((size, size))
    impulse[size // 2, size // 2] = 1
    measured = [
        np.real(np.fft.fft2(np.fft.ifftshift(element)))
        for element in hessian(impulse, sigma, mode='constant')
    ]
    omega = 2 * np.pi * np.fft.fftfreq(size)
    wr, wc = np.meshgrid(omega, omega, indexing='ij')
    damp = np.exp(-(sigma**2) * (wr**2 + wc**2) / 2)
    exact = [-(wr**2) * damp, -wr * wc * damp, -(wc**2) * damp]
    peak = np.abs(exact[0]).max()
    worst = np.max([np.abs(m - e) for m, e in zip(measured, exact)], axis=0)
    in_band = np.hypot(wr, wc) <= band * np.pi
    return worst[in_band].max() / peak


# --- Tests ----------------------------------------------------------------


def test_gain_and_shape_error_metric():
    """The metric is zero for a match, all gain for a scaling, and weights Hrc twice."""
    rng = np.random.default_rng(0)
    truth = [rng.normal(size=(64, 64)) for _ in range(3)]

    gain_error, shape_error = gain_and_shape_error(truth, truth, 1.0)
    assert gain_error == pytest.approx(0.0)
    assert shape_error == pytest.approx(0.0)

    gain_error, shape_error = gain_and_shape_error(
        [1.01 * h for h in truth], truth, 1.0
    )
    assert gain_error == pytest.approx(0.01)
    assert shape_error == pytest.approx(0.0, abs=1e-12)

    # got is orthogonal to truth, so g = 0. got is all Hrc, which counts
    # twice, so S = √2, not 1.
    zeros = np.zeros((64, 64))
    truth = [np.ones((64, 64)), zeros, zeros]
    got = [zeros, np.ones((64, 64)), zeros]
    gain_error, shape_error = gain_and_shape_error(got, truth, 1.0)
    assert gain_error == pytest.approx(-1.0)
    assert shape_error == pytest.approx(np.sqrt(2.0))


@pytest.mark.parametrize('name', ['camera', 'coins', 'astronaut'])
@pytest.mark.parametrize(
    ('sigma', 'floor'),
    [
        pytest.param(1.0, 1e-3, id='sigma 1'),
        pytest.param(2.0, 2e-12, id='sigma 2'),
        pytest.param(4.0, 2e-12, id='sigma 4'),
    ],
)
def test_matches_mobahi_rbf_photographs(fitted_sums, name, sigma, floor):
    """Largest element error against a smoothed Mobahi RBF sum.

    At sigma 1 the measured error is about ``3e-4``. At sigma 2 and above it is
    rounding, below ``1e-12``.
    """
    levels, image = fitted_sums[name]
    truth = exact_hessian(levels, image.shape, sigma)
    assert max_relative_error(hessian(image, sigma), truth, sigma) < floor


@pytest.mark.parametrize('name', ['camera', 'coins', 'astronaut'])
@pytest.mark.parametrize(
    ('sigma', 'gain_floor', 'shape_floor'),
    [
        pytest.param(1.0, 2e-4, 5e-4, id='sigma 1'),
        pytest.param(2.0, 2e-12, 2e-12, id='sigma 2'),
        pytest.param(4.0, 2e-12, 2e-12, id='sigma 4'),
    ],
)
def test_rms_matches_mobahi_rbf_photographs(
    fitted_sums, name, sigma, gain_floor, shape_floor
):
    """Whole-Hessian RMS error against a smoothed Mobahi RBF sum, by part.

    A gain error leaves the eigenvalue ratios alone, so the gain and shape
    parts of `gain_and_shape_error` fail for different reasons, and each has
    its own floor. At sigma 1 the measured parts are about ``7e-5`` (gain) and
    ``2e-4`` (shape) on all three photographs. At sigma 2 and above both are
    rounding, below ``1e-12``.
    """
    levels, image = fitted_sums[name]
    truth = exact_hessian(levels, image.shape, sigma)
    gain_error, shape_error = gain_and_shape_error(hessian(image, sigma), truth, sigma)
    assert abs(gain_error) < gain_floor
    assert shape_error < shape_floor


@pytest.mark.parametrize(
    ('sigma', 'band', 'floor'),
    [
        pytest.param(1.0, 0.5, 1e-3, id='sigma 1, half of Nyquist'),
        pytest.param(2.0, 1.0, 1e-6, id='sigma 2, up to Nyquist'),
        pytest.param(4.0, 1.0, 2e-12, id='sigma 4, up to Nyquist'),
    ],
)
def test_matches_lindeberg_frequency_response(sigma, band, floor):
    """Frequency response within ``floor`` of Lindeberg's, for ``|ω| ≤ band·π``.

    Sampling the continuous kernel aliases high frequencies, and the error
    grows toward Nyquist. At sigma 1 the test stops at half Nyquist, where the
    measured error is about ``4e-4``; over the full band it is about ``9e-2``.
    At sigma 2 and 4 the test runs to Nyquist, with errors of about ``1e-7``
    and ``1e-12``. At sigma 4 the exact response near Nyquist is numerically
    zero, so there the test bounds spurious response, not relative accuracy.
    """
    assert frequency_error(sigma, band) < floor
