"""One-pass Hessian against the two older schemes.

``hessian_matrix(..., use_gaussian_derivatives=True)`` filters each axis once,
with a sampled Gaussian derivative whose discrete moments are restored.

Two older schemes compute the same continuous derivatives:

* Two passes. Each element is a first derivative of a first derivative, at
  width ``sigma / sqrt(2)``. This is the Hessian scikit-image computed through
  version 0.26. ``two_pass`` below is that filter.
* Finite differences. Smooth, then ``np.gradient`` twice. This is
  ``use_gaussian_derivatives=False``.

The first standard is a sum of isotropic Gaussians fitted to a photograph
(``camera``, ``coins``, or greyscale ``astronaut``). Smoothing one of those
Gaussians gives another Gaussian, so the sum has an exact Hessian at every
scale. The heights come from the model in the workbook
``on_gaussian_sum_images``.

The second standard is the frequency response. The exact smoothed Hessian of
``cos(ω · x)`` is ``-ω ωᵀ exp(-σ²|ω|²/2) cos(ω · x)``. One impulse gives every
frequency at once. The photographs carry little energy near the Nyquist
frequency, so the impulse is what shows the error there.

Each floor sits just above the one-pass residue on that picture. Finite
differences stay a large fraction of the peak at every scale: two central
differences have response ``-sin²ω`` where the derivative has ``-ω²``. Two
passes lose most of the curvature at ``sigma`` around 1 and below. At
``sigma`` of 4 both sampled filters are rounding error, and two passes can be
the closer of the two.
"""

from dataclasses import dataclass

import numpy as np
import pytest
from scipy import ndimage as ndi

from _skimage2 import data, img_as_float
from _skimage2.color import rgb2gray
from _skimage2.feature import hessian_matrix


# --- The three schemes ----------------------------------------------------


def one_pass(image, sigma, mode='nearest', cval=0):
    """The installed filter: ``use_gaussian_derivatives=True``."""
    return hessian_matrix(
        image,
        sigma=sigma,
        mode=mode,
        cval=cval,
        use_gaussian_derivatives=True,
    )


def two_pass(image, sigma, mode='nearest', cval=0):
    """Two first-derivative Gaussian filters, each of width ``sigma / sqrt(2)``.

    2-D form of the Hessian scikit-image computed through version 0.26.
    The kernel runs to 8 sigma when sigma is above 1, and to 100 sigma
    otherwise. Elements come back as ``[Hrr, Hrc, Hcc]``. ``Hrc`` differentiates
    along rows first, then along columns.
    """
    image = np.asarray(image, dtype=float)
    truncate = 8 if sigma > 1 else 100
    smooth = {
        'sigma': sigma / np.sqrt(2),
        'mode': mode,
        'cval': cval,
        'truncate': truncate,
    }
    along_row = ndi.gaussian_filter(image, order=[1, 0], **smooth)
    along_col = ndi.gaussian_filter(image, order=[0, 1], **smooth)
    return [
        ndi.gaussian_filter(along_row, order=[1, 0], **smooth),
        ndi.gaussian_filter(along_row, order=[0, 1], **smooth),
        ndi.gaussian_filter(along_col, order=[0, 1], **smooth),
    ]


def finite_differences(image, sigma, mode='nearest', cval=0):
    """The installed filter: smooth at 4 sigma, then ``np.gradient`` twice."""
    return hessian_matrix(
        image,
        sigma=sigma,
        mode=mode,
        cval=cval,
        use_gaussian_derivatives=False,
    )


# --- Gaussian sums fitted to photographs ----------------------------------
#
# f(r, c) = sum_k h_k exp(-((r - r_k)**2 + (c - c_k)**2) / (2 w_k**2))
#
# Gaussians of one width form a level: a grid of centres, spacing equal to
# the width, reaching one spacing past each edge. Levels are fitted coarse
# to fine, each to the residual the coarser levels leave. Smoothing a level
# of width w by sigma replaces w with sqrt(w**2 + sigma**2) and multiplies
# the heights by w**2 / (w**2 + sigma**2).

WIDTHS = (16.0, 8.0, 4.0, 2.0, 1.5)  # pixels, coarse to fine
RIDGE = 1e-3


@dataclass
class Level:
    """Gaussians of one width, on a grid of centres."""

    width: float
    row_centres: np.ndarray
    col_centres: np.ndarray
    heights: np.ndarray  # shape (len(row_centres), len(col_centres))


def _grid_centres(n, spacing):
    """Centres along an axis of length ``n``, one spacing past each edge."""
    count = int(np.ceil((n - 1 + 2 * spacing) / spacing)) + 1
    return -spacing + spacing * np.arange(count)


def _basis(coords, centres, width, order=0):
    """1-D Gaussians at ``centres``, or a derivative, sampled at ``coords``."""
    offset = np.asarray(coords, dtype=float)[:, None] - centres[None, :]
    gaussian = np.exp(-(offset**2) / (2 * width**2))
    if order == 0:
        return gaussian
    if order == 1:
        return -offset / width**2 * gaussian
    if order == 2:
        return (offset**2 - width**2) / width**4 * gaussian
    raise ValueError(f'order must be 0, 1 or 2, not {order}')


def _ridge_pinv(matrix, ridge):
    """Tikhonov pseudo-inverse.

    ``ridge`` is a fraction of the square of the largest singular value.
    """
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    gain = singular / (singular**2 + ridge * singular[0] ** 2)
    return (right.T * gain) @ left.T


def fit(image, widths=WIDTHS, ridge=RIDGE):
    """Fit levels coarse to fine. Return the levels and the residual."""
    residual = np.array(image, dtype=float)
    rows = np.arange(residual.shape[0], dtype=float)
    cols = np.arange(residual.shape[1], dtype=float)
    levels = []
    for width in widths:
        row_centres = _grid_centres(len(rows), width)
        col_centres = _grid_centres(len(cols), width)
        row_basis = _basis(rows, row_centres, width)
        col_basis = _basis(cols, col_centres, width)
        # Separable least squares: row factor, then column factor.
        row_fit = _ridge_pinv(row_basis, ridge)
        col_fit = _ridge_pinv(col_basis, ridge)
        heights = row_fit @ residual @ col_fit.T
        residual = residual - row_basis @ heights @ col_basis.T
        levels.append(Level(width, row_centres, col_centres, heights))
    return levels, residual


def render(levels, shape, sigma=0.0, order=(0, 0)):
    """The sum smoothed by ``sigma``, or one derivative of that sum.

    ``order`` is ``(row derivative, column derivative)``. Values are at
    pixel centres, which sit on the integers.
    """
    rows = np.arange(shape[0], dtype=float)
    cols = np.arange(shape[1], dtype=float)
    out = np.zeros((len(rows), len(cols)))
    for level in levels:
        variance = level.width**2 + sigma**2
        width = np.sqrt(variance)
        scale = level.width**2 / variance
        out += scale * (
            _basis(rows, level.row_centres, width, order[0])
            @ level.heights
            @ _basis(cols, level.col_centres, width, order[1]).T
        )
    return out


def exact_hessian(levels, shape, sigma):
    """``[Hrr, Hrc, Hcc]`` of the sum smoothed by ``sigma``."""
    return [
        render(levels, shape, sigma, order=(2, 0)),
        render(levels, shape, sigma, order=(1, 1)),
        render(levels, shape, sigma, order=(0, 2)),
    ]


@pytest.fixture(scope='module')
def fitted_sums():
    """``(levels, shape)`` for ``camera``, ``coins``, and greyscale ``astronaut``."""
    images = {
        'camera': img_as_float(data.camera()),
        'coins': img_as_float(data.coins()),
        'astronaut': img_as_float(rgb2gray(data.astronaut())),
    }
    return {name: (fit(image)[0], image.shape) for name, image in images.items()}


# --- How far a scheme is from a standard ----------------------------------


def relative_error(got, truth, sigma):
    """Largest gap inside the frame, divided by the peak of ``truth``.

    The crop drops pixels where an 8-sigma kernel would read past the frame.
    There the border rule invents samples the Gaussian sum does not have.
    Two passes use a kernel of 100 sigma when sigma is 1 or less, so this
    crop still contains part of that longer border.
    """
    reach = int(8 * sigma + 0.5) + 1
    inner = (slice(reach, -reach),) * 2
    peak = max(np.abs(part[inner]).max() for part in truth)
    gaps = (
        np.abs(part[inner] - reference[inner]).max()
        for part, reference in zip(got, truth)
    )
    return max(gaps) / peak


def frequency_error(method, sigma, band, size=257):
    """Worst relative error for ``|ω| ≤ band × π``, over every orientation.

    ``band`` is a fraction of the Nyquist frequency. The impulse sits on the
    centre pixel of an odd frame; ``mode='constant'`` puts zeros outside it.
    257 is wide enough for the two-pass kernel at sigma 1, which reaches
    100 pixels. ``ifftshift`` moves the centre pixel to the corner, where
    ``fft2`` reads the origin. The imaginary part is rounding, because every
    kernel is even or odd.
    """
    impulse = np.zeros((size, size))
    impulse[size // 2, size // 2] = 1
    measured = [
        np.real(np.fft.fft2(np.fft.ifftshift(element)))
        for element in method(impulse, sigma, mode='constant')
    ]
    omega = 2 * np.pi * np.fft.fftfreq(size)
    wr, wc = np.meshgrid(omega, omega, indexing='ij')
    envelope = np.exp(-(sigma**2) * (wr**2 + wc**2) / 2)
    exact = [-(wr**2) * envelope, -wr * wc * envelope, -(wc**2) * envelope]
    peak = np.abs(exact[0]).max()
    worst = np.max([np.abs(got - truth) for got, truth in zip(measured, exact)], axis=0)
    in_band = np.hypot(wr, wc) <= band * np.pi
    return worst[in_band].max() / peak


# At this scale and above, one pass and two passes are both rounding error,
# and two passes can be the closer of the two. Finite differences stay a
# large fraction of the peak.
ROUNDING_SCALE = 4


# --- Tests ----------------------------------------------------------------


@pytest.mark.parametrize('name', ['camera', 'coins', 'astronaut'])
@pytest.mark.parametrize(
    ('sigma', 'floor'),
    [
        pytest.param(0.5, None, id='sigma 0.5, aliasing'),
        pytest.param(1.0, 1e-3, id='sigma 1'),
        pytest.param(2.0, 1e-10, id='sigma 2'),
        pytest.param(4.0, 1e-10, id='sigma 4, rounding'),
    ],
)
def test_one_pass_is_closer_on_a_fitted_gaussian_sum(fitted_sums, name, sigma, floor):
    """One pass stays near a fitted Gaussian sum, and closer than the older schemes.

    The photograph only chooses the heights. The standard is the sum, which
    has a closed-form Hessian. ``floor`` is omitted at sigma 0.5, where
    sampling the kernel aliases and the one-pass residue is a few percent.
    """
    levels, shape = fitted_sums[name]
    image = render(levels, shape)
    truth = exact_hessian(levels, shape, sigma)

    def error(method):
        return relative_error(method(image, sigma), truth, sigma)

    one = error(one_pass)
    if floor is not None:
        assert one < floor
    assert one < error(finite_differences)
    if sigma < ROUNDING_SCALE:
        assert one < error(two_pass)


@pytest.mark.parametrize(
    ('sigma', 'band', 'floor'),
    [
        pytest.param(1.0, 0.5, 1e-3, id='sigma 1, half of Nyquist'),
        pytest.param(2.0, 1.0, 1e-6, id='sigma 2, up to Nyquist'),
        pytest.param(4.0, 1.0, 1e-10, id='sigma 4, up to Nyquist'),
    ],
)
def test_one_pass_frequency_response_is_closer(sigma, band, floor):
    """One pass stays near the ideal response, and closer than the older schemes.

    ``band`` is a fraction of the Nyquist frequency. At sigma 1 the gate is the
    lower half of the band, where the one-pass alias is about ``4e-4``. At
    sigma 2 and 4 the gate is the whole band up to Nyquist.
    """
    one = frequency_error(one_pass, sigma, band)
    assert one < floor
    assert one < frequency_error(finite_differences, sigma, band)
    if sigma < ROUNDING_SCALE:
        assert one < frequency_error(two_pass, sigma, band)
