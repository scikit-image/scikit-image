import numpy as np
import pytest
import scipy.ndimage as ndi
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
    assert_array_less,
    assert_equal,
)

from _skimage2 import img_as_float
from _skimage2._shared.utils import _supported_float_type
from _skimage2.color import rgb2gray
from _skimage2.data import camera, retina
from _skimage2.feature import hessian_matrix, hessian_matrix_eigvals
from _skimage2.filters import frangi, hessian, meijering, sato
from _skimage2.filters.ridges import _frangi_norm, _scaled_hessian_frob
from _skimage2.util import crop, invert


def test_2d_null_matrix():
    a_black = np.zeros((3, 3)).astype(np.uint8)
    a_white = invert(a_black)

    zeros = np.zeros((3, 3))
    ones = np.ones((3, 3))

    assert_equal(meijering(a_black, black_ridges=True), zeros)
    assert_equal(meijering(a_white, black_ridges=False), zeros)

    assert_equal(sato(a_black, black_ridges=True, mode='reflect'), zeros)
    assert_equal(sato(a_white, black_ridges=False, mode='reflect'), zeros)

    assert_allclose(frangi(a_black, black_ridges=True), zeros, atol=1e-3)
    assert_allclose(frangi(a_white, black_ridges=False), zeros, atol=1e-3)

    assert_equal(hessian(a_black, black_ridges=False, mode='reflect'), ones)
    assert_equal(hessian(a_white, black_ridges=True, mode='reflect'), ones)


def test_3d_null_matrix():
    # Note: last axis intentionally not size 3 to avoid 2D+RGB autodetection
    #       warning from an internal call to `skimage.filters.gaussian`.
    a_black = np.zeros((3, 3, 5)).astype(np.uint8)
    a_white = invert(a_black)

    zeros = np.zeros((3, 3, 5))
    ones = np.ones((3, 3, 5))

    assert_allclose(meijering(a_black, black_ridges=True), zeros, atol=1e-1)
    assert_allclose(meijering(a_white, black_ridges=False), zeros, atol=1e-1)

    assert_equal(sato(a_black, black_ridges=True, mode='reflect'), zeros)
    assert_equal(sato(a_white, black_ridges=False, mode='reflect'), zeros)

    assert_allclose(frangi(a_black, black_ridges=True), zeros, atol=1e-3)
    assert_allclose(frangi(a_white, black_ridges=False), zeros, atol=1e-3)

    assert_equal(hessian(a_black, black_ridges=False, mode='reflect'), ones)
    assert_equal(hessian(a_white, black_ridges=True, mode='reflect'), ones)


def test_2d_energy_decrease():
    a_black = np.zeros((5, 5)).astype(np.uint8)
    a_black[2, 2] = 255
    a_white = invert(a_black)

    assert_array_less(meijering(a_black, black_ridges=True).std(), a_black.std())
    assert_array_less(meijering(a_white, black_ridges=False).std(), a_white.std())

    assert_array_less(
        sato(a_black, black_ridges=True, mode='reflect').std(), a_black.std()
    )
    assert_array_less(
        sato(a_white, black_ridges=False, mode='reflect').std(), a_white.std()
    )

    assert_array_less(frangi(a_black, black_ridges=True).std(), a_black.std())
    assert_array_less(frangi(a_white, black_ridges=False).std(), a_white.std())

    assert_array_less(
        hessian(a_black, black_ridges=True, mode='reflect').std(), a_black.std()
    )
    assert_array_less(
        hessian(a_white, black_ridges=False, mode='reflect').std(), a_white.std()
    )


def test_3d_energy_decrease():
    a_black = np.zeros((5, 5, 5)).astype(np.uint8)
    a_black[2, 2, 2] = 255
    a_white = invert(a_black)

    assert_array_less(meijering(a_black, black_ridges=True).std(), a_black.std())
    assert_array_less(meijering(a_white, black_ridges=False).std(), a_white.std())

    assert_array_less(
        sato(a_black, black_ridges=True, mode='reflect').std(), a_black.std()
    )
    assert_array_less(
        sato(a_white, black_ridges=False, mode='reflect').std(), a_white.std()
    )

    assert_array_less(frangi(a_black, black_ridges=True).std(), a_black.std())
    assert_array_less(frangi(a_white, black_ridges=False).std(), a_white.std())

    assert_array_less(
        hessian(a_black, black_ridges=True, mode='reflect').std(), a_black.std()
    )
    assert_array_less(
        hessian(a_white, black_ridges=False, mode='reflect').std(), a_white.std()
    )


def test_2d_linearity():
    a_black = np.ones((3, 3)).astype(np.uint8)
    a_white = invert(a_black)

    assert_allclose(
        meijering(1 * a_black, black_ridges=True),
        meijering(10 * a_black, black_ridges=True),
        atol=1e-3,
    )
    assert_allclose(
        meijering(1 * a_white, black_ridges=False),
        meijering(10 * a_white, black_ridges=False),
        atol=1e-3,
    )

    assert_allclose(
        sato(1 * a_black, black_ridges=True, mode='reflect'),
        sato(10 * a_black, black_ridges=True, mode='reflect'),
        atol=1e-3,
    )
    assert_allclose(
        sato(1 * a_white, black_ridges=False, mode='reflect'),
        sato(10 * a_white, black_ridges=False, mode='reflect'),
        atol=1e-3,
    )

    assert_allclose(
        frangi(1 * a_black, black_ridges=True),
        frangi(10 * a_black, black_ridges=True),
        atol=1e-3,
    )
    assert_allclose(
        frangi(1 * a_white, black_ridges=False),
        frangi(10 * a_white, black_ridges=False),
        atol=1e-3,
    )

    assert_allclose(
        hessian(1 * a_black, black_ridges=True, mode='reflect'),
        hessian(10 * a_black, black_ridges=True, mode='reflect'),
        atol=1e-3,
    )
    assert_allclose(
        hessian(1 * a_white, black_ridges=False, mode='reflect'),
        hessian(10 * a_white, black_ridges=False, mode='reflect'),
        atol=1e-3,
    )


def test_3d_linearity():
    # Note: last axis intentionally not size 3 to avoid 2D+RGB autodetection
    #       warning from an internal call to `skimage.filters.gaussian`.
    a_black = np.ones((3, 3, 5)).astype(np.uint8)
    a_white = invert(a_black)

    assert_allclose(
        meijering(1 * a_black, black_ridges=True),
        meijering(10 * a_black, black_ridges=True),
        atol=1e-3,
    )
    assert_allclose(
        meijering(1 * a_white, black_ridges=False),
        meijering(10 * a_white, black_ridges=False),
        atol=1e-3,
    )

    assert_allclose(
        sato(1 * a_black, black_ridges=True, mode='reflect'),
        sato(10 * a_black, black_ridges=True, mode='reflect'),
        atol=1e-3,
    )
    assert_allclose(
        sato(1 * a_white, black_ridges=False, mode='reflect'),
        sato(10 * a_white, black_ridges=False, mode='reflect'),
        atol=1e-3,
    )

    assert_allclose(
        frangi(1 * a_black, black_ridges=True),
        frangi(10 * a_black, black_ridges=True),
        atol=1e-3,
    )
    assert_allclose(
        frangi(1 * a_white, black_ridges=False),
        frangi(10 * a_white, black_ridges=False),
        atol=1e-3,
    )

    assert_allclose(
        hessian(1 * a_black, black_ridges=True, mode='reflect'),
        hessian(10 * a_black, black_ridges=True, mode='reflect'),
        atol=1e-3,
    )
    assert_allclose(
        hessian(1 * a_white, black_ridges=False, mode='reflect'),
        hessian(10 * a_white, black_ridges=False, mode='reflect'),
        atol=1e-3,
    )


def test_2d_cropped_camera_image():
    a_black = crop(camera(), ((200, 212), (100, 312)))
    a_white = invert(a_black)

    np.zeros((100, 100))
    ones = np.ones((100, 100))

    assert_allclose(
        meijering(a_black, black_ridges=True), meijering(a_white, black_ridges=False)
    )

    assert_allclose(
        sato(a_black, black_ridges=True, mode='reflect'),
        sato(a_white, black_ridges=False, mode='reflect'),
    )

    assert_allclose(
        frangi(a_black, black_ridges=True), frangi(a_white, black_ridges=False)
    )

    assert_allclose(
        hessian(a_black, black_ridges=True, mode='reflect'), ones, atol=1 - 1e-7
    )
    assert_allclose(
        hessian(a_white, black_ridges=False, mode='reflect'), ones, atol=1 - 1e-7
    )


@pytest.mark.parametrize('func', [meijering, sato, frangi, hessian])
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_ridge_output_dtype(func, dtype):
    img = img_as_float(camera()).astype(dtype, copy=False)
    assert func(img).dtype == _supported_float_type(img.dtype)


def test_3d_cropped_camera_image():
    a_black = crop(camera(), ((200, 212), (100, 312)))
    a_black = np.stack([a_black] * 5, axis=-1)
    a_white = invert(a_black)

    np.zeros(a_black.shape)
    ones = np.ones(a_black.shape)

    assert_allclose(
        meijering(a_black, black_ridges=True), meijering(a_white, black_ridges=False)
    )

    assert_allclose(
        sato(a_black, black_ridges=True, mode='reflect'),
        sato(a_white, black_ridges=False, mode='reflect'),
    )

    assert_allclose(
        frangi(a_black, black_ridges=True), frangi(a_white, black_ridges=False)
    )

    assert_allclose(
        hessian(a_black, black_ridges=True, mode='reflect'), ones, atol=1 - 1e-7
    )
    assert_allclose(
        hessian(a_white, black_ridges=False, mode='reflect'), ones, atol=1 - 1e-7
    )


@pytest.mark.parametrize(
    'func, tol', [(frangi, 1e-2), (meijering, 1e-2), (sato, 2e-3), (hessian, 2e-2)]
)
def test_border_management(func, tol):
    img = rgb2gray(retina()[300:500, 700:900])
    out = func(img, sigmas=[1], mode='reflect')

    full_std = out.std()
    full_mean = out.mean()
    inside_std = out[4:-4, 4:-4].std()
    inside_mean = out[4:-4, 4:-4].mean()
    border_std = np.stack([out[:4, :], out[-4:, :], out[:, :4].T, out[:, -4:].T]).std()
    border_mean = np.stack(
        [out[:4, :], out[-4:, :], out[:, :4].T, out[:, -4:].T]
    ).mean()

    assert abs(full_std - inside_std) < tol
    assert abs(full_std - border_std) < tol
    assert abs(inside_std - border_std) < tol
    assert abs(full_mean - inside_mean) < tol
    assert abs(full_mean - border_mean) < tol
    assert abs(inside_mean - border_mean) < tol


# `frangi` and scales
#
# When correct, Frangi should normalize vesselness scores across smoothing
# levels, so wide vessels are not penalized by the smoothing.  Otherwise, the
# responses of a ridge fall monotonically with greater smoothing, so the
# smallest scale always wins.  gh-7711 reported that `frangi` lost this
# normalization in version 0.20.0.  The tests below assert we do now find
# vessel-like features across scales.
#
# Every constant below is measured in the workbooks notebook
# https://scikit-image.org/skimage-workbooks/frangi-testing, which shows what
# each test catches and which of these an unrepaired `frangi` fails.


@pytest.mark.parametrize('shape', [(15, 17), (9, 10, 11)])
@pytest.mark.parametrize('sigma', [1.0, 2.5])
def test_scaled_hessian_frob_matches_eigenvalue_form(shape, sigma):
    """`_scaled_hessian_frob` gives `S` as eq. (12) of Frangi et al. (1998).

    Eq. (12) takes the root of the summed squared Hessian eigenvalues.  The
    helper instead sums the squared elements of the upper triangle that
    `hessian_matrix` returns, counting off-diagonal elements twice.  The two
    agree on a symmetric matrix only if that weighting is right.
    """
    rng = np.random.default_rng(1998)
    image = rng.random(shape)
    H_elems = hessian_matrix(image, sigma, use_gaussian_derivatives=True)
    eigvals = hessian_matrix_eigvals(H_elems)
    expected = sigma**2 * np.sqrt(np.sum(eigvals**2, axis=0))
    assert_allclose(
        _scaled_hessian_frob(H_elems, sigma), expected, rtol=1e-10, atol=1e-12
    )


def _get_default_gamma(image, sigmas, mode='reflect', cval=0):
    """Half largest Hessian norm over `sigmas`: the value `gamma=None` gives."""
    maxes = []
    for sigma in sigmas:
        maxes.append(_frangi_norm(image, sigma, mode, cval).max())
    return max(maxes) / 2


FRANGI_WIDTHS = (1.0, 2.0, 4.0, 8.0)
FRANGI_CENTRES = (28, 75, 122, 172)
FRANGI_SIGMAS = (1, 2, 3, 4, 6, 8, 10, 12)


@pytest.fixture
def frangi_bars():
    """Dark bars of four widths on a light ground, and their centre probes.

    A scale-normalised ridge filter must score these different widths alike,
    and must find each with a filter of matching size.
    """
    size = 200
    columns = np.indices((size, size), dtype=float)[1]
    image = np.ones((size, size))
    # Add the bars to the image.
    probes = []  # Example coordinate for bar in image.
    for width, centre in zip(FRANGI_WIDTHS, FRANGI_CENTRES):
        image -= np.exp(-((columns - centre) ** 2) / (2 * width**2))
        probes.append((size // 2, centre))
    return image, probes


def test_frangi_score_does_not_depend_on_bar_width(frangi_bars):
    """Four bars of the same contrast but different widths must score alike."""
    image, probes = frangi_bars
    scores = [frangi(image, sigmas=FRANGI_SIGMAS)[probe] for probe in probes]
    # Without the sigma ** 2 correction scores span a factor of 220.
    assert min(scores) / max(scores) > 0.9


def test_frangi_ignores_the_order_of_sigmas(frangi_bars):
    """`sigmas` is a set of scales, so its order must not matter."""
    image, _ = frangi_bars
    reference = frangi(image, sigmas=(1, 3, 5))
    for order in ((1, 5, 3), (3, 1, 5), (3, 5, 1), (5, 1, 3), (5, 3, 1)):
        assert_array_equal(frangi(image, sigmas=order), reference)


def test_frangi_gamma_is_resolved_over_every_scale(frangi_bars):
    """`gamma=None` is half the largest Hessian norm over all scales given.

    Passing that value explicitly must reproduce the default exactly.  It is
    the value the docstring describes.
    """
    image, _ = frangi_bars
    sigmas = (1, 3, 5)
    resolved = _get_default_gamma(image, sigmas)
    # Test passing calculated gamma explicitly gives same answer as default.
    assert_array_equal(
        frangi(image, sigmas=sigmas), frangi(image, sigmas=sigmas, gamma=resolved)
    )


def test_frangi_agrees_with_sato_about_scale(frangi_bars):
    """Sato and Frangi must nominate the same scale for the same bar.

    We can only compare results of passing single sigmas with `gamma` held
    fixed. Otherwise, with `gamma=None`, gamma resolves to half the largest
    Hessian norm *of the scales it was given*, so a call carrying one sigma
    takes `gamma` from that sigma alone. For a single sigma, default `gamma` is
    `S.max() / 2`, so `S.max() / gamma` is 2 by construction, and the
    structuredness term `1 - exp(-S ** 2 / 2 gamma ** 2)` is `1 - exp(-2)` at
    the strongest pixel of every scale.  As a result, results from different
    single sigmas are not comparable unless `gamma` is supplied explicitly.

    Not every fixed value of `gamma` is useful.  A `gamma` far below the
    Hessian norms saturates the structuredness term to 1 at every scale, and
    the scan across sigmas then picks a scale from blobness alone.  We use the
    value that `gamma=None` gives when using the `FRANGI_SIGMAS` values.
    """
    image, probes = frangi_bars
    # Fix Frangi gamma for repeat calls across sigmas.
    gamma = _get_default_gamma(image, FRANGI_SIGMAS)
    for probe in probes:
        sato_scales = []
        frangi_scales = []
        for s in FRANGI_SIGMAS:
            sato_scales.append(sato(image, sigmas=[s], mode='reflect')[probe])
            frangi_scales.append(
                frangi(image, sigmas=[s], gamma=gamma, mode='reflect')[probe]
            )
        # They identify the same scale.
        assert np.argmax(sato_scales) == np.argmax(frangi_scales)


@pytest.mark.parametrize('width', FRANGI_WIDTHS)
def test_frangi_rejects_the_wrong_polarity(width):
    """`black_ridges=True` must say nothing about a bright ridge.

    `frangi` negates the image for `black_ridges=False`, so internally it
    always looks for dark ridges.  Across a dark ridge, the curvature is
    positive.  After sorting the Hessian eigenvalues by magnitude, `lambda1`
    is the small value along the ridge, and `lambda2` is the larger value
    across the ridge.  Eq. (15) of Frangi et al. (1998) gives zero vesselness
    unless `lambda2 > 0`.

    In 2D, the blobness ratio R_B is ``|lambda1| / lambda2``.  It is high for
    a blob, where the two eigenvalues are similar, and low for a ridge.  The
    blobness factor ``exp(-R_B ** 2 / (2 * beta ** 2))`` is near 1 for a ridge
    and falls towards 0 for a blob.

    In previous code, we did this::

        lambda2 = np.maximum(eigvals[1], 1e-10)
        r_b = abs(lambda1) / lambda2
        vals *= np.exp(-r_b**2 / (2 * beta**2))   # blobness

    The code did not check the sign of `lambda2` directly.  Instead it relied
    on clipping: a wrong-signed `lambda2` becomes 1e-10, so `r_b` becomes very
    large, and ``np.exp(-r_b**2 / (2 * beta**2))`` underflows to 0.  This
    works when `lambda1` is not very close to zero (roughly, above 1e-9 with
    the default ``beta=0.5``), as will usually be the case in a real image.

    Now consider a positive (bright) ridge when `black_ridges=True` (the
    default), so the image has not been negated.  The ideal bright ridge has
    ``lambda1 = 0`` and ``lambda2 < 0``.  In the code above, ``r_b = 0 / 1e-10
    = 0``, so the blobness factor is 1, rather than 0.  The structuredness
    factor uses the unclipped eigenvalues, so it is also large, and the bright
    ridge gets a high score.  The current code tests the sign of `lambda2`
    explicitly, and gives 0 wherever the sign does not match the `black_ridges`
    parameter.
    """
    # An ideal bright ridge.
    size = 128
    columns = np.indices((size, size), dtype=float)[1]
    bright = np.exp(-((columns - size // 2) ** 2) / (2 * width**2))
    centre = (size // 2, size // 2)

    # Default is black_ridges=True; it should not detect the ideal bright
    # ridge.
    assert frangi(bright, sigmas=[3])[centre] == 0
    # But it should detect the same ridge when looking for bright ridges, so
    # the zero above is not just a filter that returns zero everywhere.
    assert frangi(bright, sigmas=[3], black_ridges=False)[centre] > 0


@pytest.mark.parametrize('factor', [1e-6, 1e-4, 1e2])
def test_frangi_is_invariant_to_brightness_units(factor):
    """With `gamma=None` every term is a ratio, so scaling must change nothing.

    The paper calls the geometric ratios "grey-level invariant".
    """
    image = img_as_float(camera())[::4, ::4]
    assert_allclose(
        frangi(image * factor, sigmas=(1, 3, 5)),
        frangi(image, sigmas=(1, 3, 5)),
        atol=1e-6,
    )


def test_frangi_alpha_has_no_effect_in_2d():
    """`alpha` is the plate factor, which eq. (15) does not have.

    Therefore, alpha should have no effect in 2D, as noted in docstring, but
    does change result in 3D.
    """
    rng = np.random.default_rng(0)
    flat = rng.random((48, 48))
    volume = rng.random((24, 24, 24))

    # Default alpha is 0.5.
    reference = frangi(flat, sigmas=(1, 3))
    # alpha has no effect in 2D.
    for alpha in (0.1, 2.0, 5.0):
        assert_array_equal(frangi(flat, sigmas=(1, 3), alpha=alpha), reference)
    # alpha has an effect in 3D.
    assert not np.allclose(
        frangi(volume, sigmas=(1, 3), alpha=0.1),
        frangi(volume, sigmas=(1, 3), alpha=5.0),
    )


def test_frangi_sigma_iterable():
    """Check sigmas can be generator."""
    rng = np.random.default_rng(0)
    img = rng.random((48, 48))
    res = frangi(img, sigmas=(1, 3))
    sigmas = (i for i in (1, 3))
    assert_array_equal(res, frangi(img, sigmas=sigmas))


# The `alpha` parameter of `meijering`.
#
# `meijering`'s alpha parameter is a tuning parameter to increase (or decrease)
# scores of line-like features, compared to blob / dot-like features.  Because
# the tuning is to shape, alpha can also be called a shaping parameter.
#
# See the original paper referenced in the `meijering` docstring for details.
#
# The tests below check that the default alpha gives higher scores for lines
# than dots, while allowing for the fact that the peak values of lines and dots
# scale differently under Gaussian smoothing.
#
# The justification for these tests is in the scikit-image workbooks notebook
# https://scikit-image.org/skimage-workbooks/meijering-alpha-testing.  Most of
# the setup here is making a dot-and-line image in which we have scaled the
# line and dot intensity so that they have matching peak heights under the
# smoothing at which we test.  This is because blurring reduces peak blob
# height more than it reduces peak line height (smoothing a blob pulls in low
# signal in all directions, whereas smoothing a line pulls in some high signal
# from the surrounding line).  Scaling the dot by ``sqrt(1 +
# sigma**2/width**2)`` cancels that, so dot and line tie exactly at ``alpha =
# 0`` and any later inequality is result of the alpha parameter.

MEIJERING_WIDTH = 4.0  # structure half-width, and the sigma we filter at.


def _dot_and_line(ndim=2, size=161, angle=0.0, width=MEIJERING_WIDTH):
    """A bright line and a bright dot that meijering scores alike at alpha=0.

    The line runs along the last axis, turned by `angle` degrees in the plane
    of the last two axes; the dot is an isotropic Gaussian of the same width.
    Returns the image and the two probe points, at their centres.
    """
    far, near = round(0.7 * size), round(0.28 * size)
    line_at = (far,) * (ndim - 1) + (near,)
    dot_at = (near,) * (ndim - 1) + (far,)

    coords = np.indices((size,) * ndim, dtype=float)
    offsets = [c - p for c, p in zip(coords, line_at)]
    direction = np.zeros(ndim)  # the direction the line is flat along
    direction[-2:] = np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))
    along = sum(d * o for d, o in zip(direction, offsets))
    across_sq = sum(o**2 for o in offsets) - along**2
    line = np.exp(-across_sq / (2 * width**2))

    amplitude = np.sqrt(2)  # sqrt(1 + sigma**2/width**2), with sigma = width
    dot_sq = sum((c - p) ** 2 for c, p in zip(coords, dot_at))
    dot = amplitude * np.exp(-dot_sq / (2 * width**2))
    return line + dot, dot_at, line_at


def _dot_line_ratio(image, dot_at, line_at, alpha, width=MEIJERING_WIDTH):
    """Ratio of meijering's score at the dot centre to that at the line."""
    out = meijering(image, sigmas=[width], alpha=alpha, black_ridges=False)
    return out[dot_at] / out[line_at]


@pytest.mark.parametrize('ndim, size', [(2, 161), (3, 65)])
def test_meijering_alpha_dot_line_scores(ndim, size):
    """Test various alphas with dot and line.

    With alpha=0, ratio is around 1.

    With default alpha, a dot scores 2/3 of a line, in 2-D and 3-D.

    See workbook for explanation of 2/3 criterion.  Note also, from workbooks,
    that these tests can't be extended to > 3D.
    """
    expected_ratio = 2 / 3  # dot to line ratio.  <1 prefers line to dot.
    def_alpha = -1 / (ndim + 1)  # Default alpha (alpha=None)
    image, dot_at, line_at = _dot_and_line(ndim, size)
    # With the shaping off (alpha=0), the two score the same.
    assert_allclose(_dot_line_ratio(image, dot_at, line_at, alpha=0.0), 1.0, rtol=1e-6)
    # Ratio should be 2 / 3 for default alpha, preferring line to dot.
    def_dot_line = _dot_line_ratio(image, dot_at, line_at, alpha=None)
    assert np.isclose(def_dot_line, expected_ratio, rtol=1e-6)
    # Check that alpha=None corresponds to stated default.
    assert_equal(_dot_line_ratio(image, dot_at, line_at, alpha=def_alpha), def_dot_line)
    # For a while we had a positive default alpha, that has the opposite to the
    # desired effect.  Assert that this (previous, positive, incorrect) default
    # gives the expected output for that alpha, but different to that above.
    # Note that positive alpha prefers the dot to the line (dot/line ratio >
    # 1).
    assert_allclose(
        _dot_line_ratio(image, dot_at, line_at, alpha=-def_alpha),
        2 * ndim / (2 * ndim - 1),  # Ratio is positive (prefers dot to line).
        rtol=1e-6,
    )

    # Apply similar tests to a random image.
    rng = np.random.default_rng(0)
    rand_image = ndi.gaussian_filter(rng.random((32,) * ndim), 1.5)
    sigmas = [2.0]
    rand_def_dot_line = meijering(rand_image, sigmas=sigmas)
    assert_array_equal(
        rand_def_dot_line, meijering(rand_image, sigmas=sigmas, alpha=def_alpha)
    )
    # Reversing the sign of alpha gives a different result.
    assert not np.allclose(
        rand_def_dot_line, meijering(rand_image, sigmas=sigmas, alpha=-def_alpha)
    )


def test_meijering_alpha_linear_between_landmarks():
    """The dot-to-line score is 1 + alpha: a tie at 0, no dot at -1.

    The relationship holds on that interval only.  See workbook.
    """
    image, dot_at, line_at = _dot_and_line()
    for alpha in np.linspace(-1.0, 1.0, 9):
        assert_allclose(
            _dot_line_ratio(image, dot_at, line_at, alpha),
            1 + alpha,
            rtol=1e-6,
            atol=1e-12,
        )


def test_meijering_alpha_ignores_orientation():
    """Rotating the line must not change what alpha does to it."""
    straight, dot_at, line_at = _dot_and_line(angle=0)
    straight_ratio = (_dot_line_ratio(straight, dot_at, line_at, alpha=None),)
    for angle in [15, 30, 45, 63, 90]:
        turned, turned_dot, turned_line = _dot_and_line(angle=angle)
        assert_allclose(
            _dot_line_ratio(turned, turned_dot, turned_line, alpha=None),
            straight_ratio,
            rtol=1e-6,
        )
