"""Accuracy of ``hessian_matrix`` against closed-form Hessians.

``hessian_matrix(..., use_gaussian_derivatives=True)`` filters each axis once,
with a sampled Gaussian derivative whose discrete moments have been restored.
These tests measure how close that discrete filter stays to two continuous
standards.

1. Mobahi's Gaussian RBF sums [1], fitted to photographs (``camera``,
   ``coins``, greyscale ``astronaut``). Gaussian smoothing maps such a sum to
   another sum of the same form, so its Hessian is exact at every scale
   (``tests/gaussian_sum.py``, used through the ``gaussian_sum_images``
   fixture). The amplitudes come from separable ridge least squares [2]; the
   photograph only chooses them. Two measures compare the filter
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

import numpy as np
import pytest

from skimage.feature import hessian_matrix

from ... import gaussian_sum


def hessian(image, sigma, mode='nearest'):
    """The installed Gaussian-derivative Hessian, ``[Hrr, Hrc, Hcc]``."""
    return hessian_matrix(image, sigma=sigma, mode=mode, use_gaussian_derivatives=True)


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
def test_matches_mobahi_rbf_photographs(gaussian_sum_images, name, sigma, floor):
    """Largest element error against a smoothed Mobahi RBF sum.

    At sigma 1 the measured error is about ``3e-4``. At sigma 2 and above it is
    rounding, below ``1e-12``.
    """
    levels, image = gaussian_sum_images[name]
    truth = gaussian_sum.hessian(levels, image.shape, sigma)
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
    gaussian_sum_images, name, sigma, gain_floor, shape_floor
):
    """Whole-Hessian RMS error against a smoothed Mobahi RBF sum, by part.

    A gain error leaves the eigenvalue ratios alone, so the gain and shape
    parts of `gain_and_shape_error` fail for different reasons, and each has
    its own floor. At sigma 1 the measured parts are about ``7e-5`` (gain) and
    ``2e-4`` (shape) on all three photographs. At sigma 2 and above both are
    rounding, below ``1e-12``.
    """
    levels, image = gaussian_sum_images[name]
    truth = gaussian_sum.hessian(levels, image.shape, sigma)
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
