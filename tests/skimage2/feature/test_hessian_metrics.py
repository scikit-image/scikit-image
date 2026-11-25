"""Accuracy of ``hessian_matrix`` against continuous gold standards.

``hessian_matrix(..., use_gaussian_derivatives=True)`` filters each axis once,
with a sampled Gaussian derivative for which discrete moments have been
restored. These tests ask how close that discrete filter stays to closed-form
continuous Hessians.

The first standard is Mobahi's isotropic Gaussian RBF sum fitted to a
photograph (``camera``, ``coins``, or greyscale ``astronaut``). With centres
``x_j``, widths ``δ_j`` and amplitudes ``a_j``,

    f(x) = Σ_j a_j exp(−‖x − x_j‖² / (2 δ_j²)),

convolution with the normalised Gaussian ``k_σ`` is closed form (Mobahi 2016,
§2.2) [1]: each term keeps its centre, replaces ``δ`` by ``√(δ² + σ²)``, and
multiplies ``a`` by ``(δ / √(δ² + σ²))^n`` with ``n = 2``. The Hessian of that
smoothed field is therefore exact at every scale. Amplitudes are fitted by
two-step separable least squares (Fausett & Fulton 1994) with a relative
ridge (Tikhonov) penalty ``RIDGE``; see
https://scikit-image.org/skimage-workbooks/on-gaussian-sum-images.
Two measures compare a filter to that standard: the largest gap of any one
element, relative to the peak, and the relative RMS error of the whole matrix
by the Frobenius norm, split into a common gain and the rest.

The second standard is Lindeberg's scale-space sinusoid (1998, §3) [3]. With
scale parameter ``t = σ²``, a one-dimensional wave ``f(x) = sin(ω₀ x)`` has
scale-space representation

    L(x; t) = exp(−ω₀² t / 2) sin(ω₀ x),

and the amplitude of the ``m``th spatial derivative is
``ω₀ᵐ exp(−ω₀² t / 2)``.
For the Hessian, ``m = 2``, so a plane wave of frequency ``ω`` has smoothed
Hessian ``−ω ωᵀ exp(−|ω|² t / 2) cos(ωᵀ x)`` (or the same factor times
``sin``). One impulse gives every frequency at once. The photographs carry
little energy near the Nyquist frequency, so the impulse is what exposes the
filter's aliasing there; the frequency tests gate whichever part of the band is
accurate at each scale.

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

from _skimage2 import data, img_as_float
from _skimage2.color import rgb2gray
from _skimage2.feature import hessian_matrix


def hessian(image, sigma, mode='nearest', cval=0):
    """The installed Gaussian-derivative Hessian."""
    return hessian_matrix(
        image,
        sigma=sigma,
        mode=mode,
        cval=cval,
        use_gaussian_derivatives=True,
    )


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

DELTAS = (16.0, 8.0, 4.0, 2.0, 1.5)  # δ, pixels, coarse to fine
RIDGE = 1e-3  # relative ridge penalty for the separable least-squares fit
NDIM = 2  # n = dim(x) for images


@dataclass
class Level:
    """Gaussians of one width δ, on a grid of centres x_j = (r_j, c_j)."""

    delta: float
    row_centres: np.ndarray
    col_centres: np.ndarray
    amplitudes: np.ndarray  # a_j; shape (len(row_centres), len(col_centres))


def _grid_centres(n, spacing):
    """Centres along an axis of length ``n``, one spacing past each edge."""
    count = int(np.ceil((n - 1 + 2 * spacing) / spacing)) + 1
    return -spacing + spacing * np.arange(count)


def _basis(coords, centres, delta, order=0):
    """1-D factors ``exp(−(t − t_j)² / (2 δ²))``, or a derivative."""
    offset = np.asarray(coords, dtype=float)[:, None] - centres[None, :]
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

    This is Tikhonov regularization with ``Γ = √λ I``, so it is ordinary least
    squares on the augmented data

        X_aug = [X; √λ I],    Y_aug = [Y; 0],

    because ``X_augᵀ X_aug = Xᵀ X + λ I`` and ``X_augᵀ Y_aug = Xᵀ Y``.
    ``npl.lstsq`` computes that solution directly.

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

    The isotropic Gaussian basis is separable: each grid term factors into a
    row Gaussian times a column Gaussian, and the rectangular centre grid
    makes the 2-D dictionary the outer product of those 1-D dictionaries.
    The model is therefore ``A_r @ A @ A_c.T ≈ residual`` with dense
    amplitudes ``A`` (Mobahi's ``a_j`` [1]). Fit the separable model by two
    successive ridge least-squares solves (Fausett & Fulton [2] for the
    separable structure): ``A_r M ≈ residual``, then ``A_c A.T ≈ M.T``.
    This is a two-step separable ridge, not the joint 2-D ridge of the
    vectorised model. Separability is in the basis, not in ``A``.

    ``ridge`` is a relative Tikhonov penalty, scaled by the largest squared
    singular value of each basis. Raising it shrinks the amplitudes and the
    cancellation between them, at the cost of fit accuracy.
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
        # Two-step separable ridge [2]: rows, then columns.
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
    ``√(δ² + σ²)`` with amplitudes
    ``a_j (δ / √(δ² + σ²))^n``. At ``σ = 0`` this is ``f`` itself.

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
    return [
        render(levels, shape, sigma, order=(2, 0)),
        render(levels, shape, sigma, order=(1, 1)),
        render(levels, shape, sigma, order=(0, 2)),
    ]


@pytest.fixture(scope='module')
def fitted_sums():
    """``(levels, shape)`` for ``camera``, ``coins``, and greyscale ``astronaut``.

    The fits are checked to stay within 10% RMS of the images. A degraded fit
    would weaken the photograph as a test image without failing the Hessian
    comparison itself, because the standard is derived from the same fit.
    """
    images = {
        'camera': img_as_float(data.camera()),
        'coins': img_as_float(data.coins()),
        'astronaut': img_as_float(rgb2gray(data.astronaut())),
    }
    fitted = {}
    for name, image in images.items():
        levels, residual = fit(image)
        relative_rms = np.sqrt(np.mean(residual**2)) / np.sqrt(np.mean(image**2))
        assert relative_rms < 0.10, (
            f'{name}: fit RMS residual is {relative_rms:.1%} of image RMS'
        )
        fitted[name] = (levels, image.shape)
    return fitted


# --- Distance to a gold standard ------------------------------------------


def _interior(sigma):
    """Pixels an 8-sigma kernel reaches without crossing the frame.

    The 2-D case of ``test_corner._interior_slice``; kept local so this
    module does not import another test module.
    """
    reach = int(8 * sigma + 0.5) + 1
    return (slice(reach, -reach),) * 2


def relative_error(got, truth, sigma):
    """Largest gap inside the frame, divided by the peak of ``truth``.

    The crop drops pixels where an 8-sigma kernel would read past the frame.
    There the border rule supplies values that differ from the Gaussian sum's.
    """
    inner = _interior(sigma)
    peak = max(np.abs(part[inner]).max() for part in truth)
    gaps = (
        np.abs(part[inner] - reference[inner]).max()
        for part, reference in zip(got, truth)
    )
    return max(gaps) / peak


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
    so ``E² = (g − 1)² + S²`` with
    ``S = ‖got − g truth‖ / ‖truth‖``. A uniform gain error scales every
    eigenvalue alike and leaves eigenvectors and eigenvalue ratios alone;
    ``S`` is the error that no one gain explains. See the RMS section of
    https://scikit-image.org/skimage-workbooks/on-gaussian-sum-images.

    Returns ``(g − 1, S)``. The crop is ``_interior(sigma)``.
    """
    inner = _interior(sigma)

    def frobenius(A, B):
        return sum(
            w * np.sum(a[inner] * b[inner]) for w, a, b in zip(FROBENIUS_WEIGHTS, A, B)
        )

    norm2 = frobenius(truth, truth)
    gain = frobenius(got, truth) / norm2
    rest = [a - gain * b for a, b in zip(got, truth)]
    return gain - 1, np.sqrt(frobenius(rest, rest) / norm2)


def frequency_error(sigma, band, size=257):
    """Worst error for ``|ω| ≤ band × π``, relative to the peak response.

    The residual is divided by the peak of the ideal ``Hrr`` transfer, so this
    is a global relative error, not the relative error at each frequency.

    Compares the filter to Lindeberg's scale-space sinusoid (1998, §3). With
    ``t = σ²``, the ``m``th Gaussian derivative of a wave of frequency ``ω``
    has amplitude ``|ω|ᵐ exp(−|ω|² t / 2)``; for the Hessian, ``m = 2``, so
    each element multiplies by ``−ω_i ω_j exp(−|ω|² t / 2)``. ``band`` is a
    fraction of the Nyquist frequency. The impulse sits on the centre pixel
    of an odd frame; ``mode='constant'`` puts zeros outside it. 257 is wide
    enough for an 8-sigma kernel at sigma 4. ``ifftshift`` moves the centre
    pixel to the corner, where ``fft2`` reads the origin. The imaginary part
    is rounding, because every kernel is even or odd.
    """
    impulse = np.zeros((size, size))
    impulse[size // 2, size // 2] = 1
    measured = [
        np.real(np.fft.fft2(np.fft.ifftshift(element)))
        for element in hessian(impulse, sigma, mode='constant')
    ]
    omega = 2 * np.pi * np.fft.fftfreq(size)
    wr, wc = np.meshgrid(omega, omega, indexing='ij')
    # Lindeberg: L(·; t) of a sinusoid damps by exp(−|ω|² t / 2), t = σ².
    t = sigma**2
    damp = np.exp(-t * (wr**2 + wc**2) / 2)
    exact = [-(wr**2) * damp, -wr * wc * damp, -(wc**2) * damp]
    peak = np.abs(exact[0]).max()
    worst = np.max([np.abs(got - truth) for got, truth in zip(measured, exact)], axis=0)
    in_band = np.hypot(wr, wc) <= band * np.pi
    return worst[in_band].max() / peak


# --- Tests ----------------------------------------------------------------


def test_gain_and_shape_error_metric():
    """The gain/shape split is exact, linear, and weights Hrc twice."""
    rng = np.random.default_rng(0)
    truth = [rng.normal(size=(64, 64)) for _ in range(3)]

    gain_error, shape = gain_and_shape_error(truth, truth, 1.0)
    assert gain_error == pytest.approx(0.0)
    assert shape == pytest.approx(0.0)

    gain_error, shape = gain_and_shape_error([1.01 * h for h in truth], truth, 1.0)
    assert gain_error == pytest.approx(0.01)
    assert shape == pytest.approx(0.0, abs=1e-12)

    # Only Hrc differs. Since it counts twice, S = sqrt(2), not 1.
    # truth has no gain, so g = 0 and g - 1 = -1.
    truth = [np.ones((64, 64)), np.zeros((64, 64)), np.zeros((64, 64))]
    got = [np.zeros((64, 64)), np.ones((64, 64)), np.zeros((64, 64))]
    gain_error, shape = gain_and_shape_error(got, truth, 1.0)
    assert gain_error == pytest.approx(-1.0)
    assert shape == pytest.approx(np.sqrt(2.0))


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
    """Hessian stays within ``floor`` of a smoothed Mobahi RBF sum.

    The photograph only chooses the amplitudes ``a_j``. The standard is
    Mobahi's smoothed sum, whose Hessian is closed form. The sigma-1 floor
    sits above the measured residue of about ``3e-4``; at sigma 2 and above
    the residue is rounding (around ``1e-12``) and the floor is ``2e-12``.
    """
    levels, shape = fitted_sums[name]
    image = render(levels, shape)
    truth = exact_hessian(levels, shape, sigma)
    assert relative_error(hessian(image, sigma), truth, sigma) < floor


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

    `gain_and_shape_error` splits the relative RMS error into a common gain
    ``g − 1`` and the rest ``S``. A gain error leaves the eigenvalue ratios
    alone, so the two parts fail for different reasons and are bounded
    separately. At sigma 1 the measured parts are about ``7e-5`` (gain) and
    ``2e-4`` (shape) on all three photographs; at sigma 2 and above both are
    rounding (around ``1e-12``) and the floors are ``2e-12``.
    """
    levels, shape = fitted_sums[name]
    image = render(levels, shape)
    truth = exact_hessian(levels, shape, sigma)
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
    """Hessian stays within ``floor`` of Lindeberg's scale-space sinusoid response.

    ``band`` is the fraction of Nyquist under test (``|ω| ≤ band·π``), and the
    error is divided by the peak of the ideal ``Hrr`` response, so it is a
    global bound. Sampling the continuous kernel aliases high frequencies; that
    residue grows toward Nyquist. At sigma 1 the check stops at half Nyquist,
    where the measured alias is about ``4e-4`` and ``floor`` is ``1e-3``; the
    full-band error rises to about ``9e-2``. At sigma 2 and 4 the check runs to
    Nyquist, with residues about ``1e-7`` and ``1e-12``. At sigma 4 the ideal
    response near Nyquist is numerically zero, so that part of the check bounds
    spurious response rather than relative accuracy.
    """
    assert frequency_error(sigma, band) < floor
