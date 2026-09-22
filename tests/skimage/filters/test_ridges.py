import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
    assert_array_less,
    assert_equal,
)

from _skimage2._shared.utils import _supported_float_type

from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.data import camera, retina
from skimage.filters import frangi, hessian, meijering, sato
from skimage.util import crop, invert

from _skimage2.filters.ridges import _frangi_shape_norm


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
# gh-7711 reported that, in version 0.20.0, we lost Frangi's standard
# normalization of vesselness scores over smoothing levels. Without the
# normalization,  the response of a ridge falls monotonically with sigma, so
# the smallest scale always wins and the filter cannot correctly select a
# structure's width.  The tests below assert we do now find vessel-like
# features across scales, as well as asserting the following two fixes:
#
# * We do now reject an ideal (noiseless) ridge with the wrong sign and
#   black_ridges = True.
# * The output previously depended on the order of passed sigmas.
#
# Every constant below is measured in the workbooks notebook
# https://scikit-image.org/skimage-workbooks/frangi_testing, which shows what
# each test catches and which of these an unrepaired `frangi` fails.


def _resolved_gamma(image, sigmas, alpha=0.5, beta=0.5, mode='reflect', cval=0):
    """Half largest Hessian norm over `sigmas`: the value `gamma=None` gives.

    Any constant held fixed across a scale scan satisfies the rule above, but
    this is the one the docstring defines, and not all values are useful: a
    `gamma` far below the Hessian norms saturates the structuredness term to 1
    at every scale, and the scan across sigmas then picks a scale from blobness
    alone.
    """
    return (
        max(
            _frangi_shape_norm(
                image, sigma, alpha=alpha, beta=beta, mode=mode, cval=cval
            )[1].max()
            for sigma in sigmas
        )
        / 2
    )


FRANGI_WIDTHS = (1.0, 2.0, 4.0, 8.0)
FRANGI_CENTRES = (28, 75, 122, 172)
FRANGI_SIGMAS = (1, 2, 3, 4, 6, 8, 10, 12)


@pytest.fixture
def frangi_bars():
    """Dark bars of four widths on a light ground, and their centre probes.

    One contrast, four widths: a scale-normalised ridge filter must score them
    alike, and must find each with a filter of matching size.
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
    """Four bars of the same contrast and four widths must score alike."""
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
    resolved = _resolved_gamma(image, sigmas)
    # Test passing calculated gamma explicitly gives same answer as default.
    assert_array_equal(
        frangi(image, sigmas=sigmas), frangi(image, sigmas=sigmas, gamma=resolved)
    )


def test_frangi_agrees_with_sato_about_scale(frangi_bars):
    """Two ridge filters must nominate the same scale for the same bar.

    We can only compare result of passing single sigmas with `gamma` held
    fixed. Otherwise, with `gamma=None`, gamma resolves to half the largest
    Hessian norm *of the scales it was given*, so a call carrying one sigma
    takes `gamma` from that sigma alone. Then `S.max() / gamma` is 2 by
    construction, and the structuredness term `1 - exp(-S ** 2 / 2 gamma ** 2)`
    is `1 - exp(-2)` at the strongest pixel of every scale.  As a result,
    results from different single sigmas are not comparable unless `gamma` is
    supplied explicitly.
    """
    image, probes = frangi_bars
    # Fix Frangi gamma for repeat calls across sigmas.
    gamma = _resolved_gamma(image, FRANGI_SIGMAS)
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

    An ideal ridge has `lambda1` exactly zero.
    """
    size = 128
    columns = np.indices((size, size), dtype=float)[1]
    bright = np.exp(-((columns - size // 2) ** 2) / (2 * width**2))
    centre = (size // 2, size // 2)

    assert frangi(bright, sigmas=[3])[centre] == 0
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

    Therefore, alpha should have no effect in 2D, as noted in docstring.
    """
    rng = np.random.default_rng(0)
    flat = rng.random((48, 48))
    volume = rng.random((24, 24, 24))

    reference = frangi(flat, sigmas=(1, 3))
    for alpha in (0.1, 2.0, 5.0):
        assert_array_equal(frangi(flat, sigmas=(1, 3), alpha=alpha), reference)
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
