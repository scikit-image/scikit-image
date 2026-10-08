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
from _skimage2.filters import frangi, hessian, meijering, sato
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


# The `alpha` parameter of `meijering`.
#
# `alpha` mixes the Hessian eigenvalues before they are scored:
# ``l'_i = l_i + alpha * sum(l_j for j != i)``.  The paper and the docstring
# set ``alpha = -1/(ndim+1)``; a sign error once shipped ``+1/(ndim+1)``.
#
# These tests assert that choice through what it does to a picture rather than
# to an eigenvalue.  Every constant below is measured in the scikit-image
# workbooks notebook
# https://scikit-image.org/skimage-workbooks/meijering-alpha-testing, which
# also derives the 2/3 and the range on which the linear law holds.  Two facts
# shape the design:
#
# * meijering divides each scale by its own maximum, so a single response
#   carries no information.  Each test asserts a ratio between two points of
#   one image, where that divisor cancels.
# * blurring costs a round dot more height than a long line.  Scaling the dot
#   by ``sqrt(1 + sigma**2/width**2)`` cancels that, so dot and line tie
#   exactly at ``alpha = 0`` and any later inequality is alpha's doing.

MEIJERING_WIDTH = 4.0  # structure half-width, and the scale we filter at.


def _dot_and_line(ndim=2, size=161, angle=0.0, width=MEIJERING_WIDTH):
    """A bright line and a bright dot that meijering scores alike at alpha=0.

    The line runs along the last axis, turned by `angle` degrees in the plane
    of the last two axes; the dot is an isotropic Gaussian of the same width.
    Returns the image and the two probe points, at its centres.
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


def _dot_over_line(image, dot_at, line_at, alpha, width=MEIJERING_WIDTH):
    """meijering's score at the dot centre, over its score at the line."""
    out = meijering(image, sigmas=[width], alpha=alpha, black_ridges=False)
    return out[dot_at] / out[line_at]


def test_meijering_alpha_ranks_a_line_above_a_dot():
    """The default must prefer an elongated feature to a blob-like one."""
    image, dot_at, line_at = _dot_and_line()
    # The picture is fair: with the shaping off (alpha=0), the two score the
    # same.
    assert_allclose(_dot_over_line(image, dot_at, line_at, alpha=0.0), 1.0, rtol=1e-6)
    # With default alpha (None), dot suppressed relative to line.
    assert _dot_over_line(image, dot_at, line_at, None) < 1.0


@pytest.mark.parametrize('ndim, size', [(2, 161), (3, 65)])
def test_meijering_alpha_scores_a_dot_at_two_thirds_of_a_line(ndim, size):
    """With the default, a dot scores exactly 2/3 of a line, in 2-D and 3-D.

    n (below) is the dimensionality of the input image (n == 2 for 2D).

    After black_ridges=False normalizes the bright line to a dark one, its ideal
    Hessian eigenvalues are (a, ..., a, 0), with a > 0.  Alpha=-1/(n+1) gives
    3a/(n+1) for each transverse direction and -(n-1)a/(n+1) for the flat
    direction.  The transverse value is selected in 2-D and 3-D; the magnitudes
    tie in 4-D; and the negative flat value is selected and clipped to zero from
    5-D onward.  Do not extend this ratio assertion past 3-D without defining
    the intended n-D behavior.  The shipped +1/(n+1) gives 2n/(2n-1), always
    above one.
    """
    image, dot_at, line_at = _dot_and_line(ndim, size)
    # Ratio should be 2 / 3 for default alpha, preferring line to dot.
    assert_allclose(
        _dot_over_line(image, dot_at, line_at, alpha=None), 2 / 3, rtol=1e-6
    )
    # For the previous, erroneous positive alpha, ratio as calulated below,
    # which is always > 1 (prefers dot).
    assert_allclose(
        _dot_over_line(image, dot_at, line_at, alpha=1 / (ndim + 1)),
        2 * ndim / (2 * ndim - 1),
        rtol=1e-6,
    )


def test_meijering_alpha_is_linear_between_its_landmarks():
    """The dot-to-line score is |1 + alpha|: a tie at 0, no dot at -1.

    Outside [-1, 1] the filter starts reading the wrong curvature, so the law
    holds on that interval only.
    """
    image, dot_at, line_at = _dot_and_line()
    for alpha in np.linspace(-1.0, 1.0, 9):
        assert_allclose(
            _dot_over_line(image, dot_at, line_at, alpha),
            abs(1 + alpha),
            rtol=1e-6,
            atol=1e-12,
        )


@pytest.mark.parametrize('angle', [15, 30, 45, 63, 90])
def test_meijering_alpha_ignores_orientation(angle):
    """Turning the picture must not change what alpha does to it."""
    straight, dot_at, line_at = _dot_and_line(angle=0)
    turned, turned_dot, turned_line = _dot_and_line(angle=angle)
    assert_allclose(
        _dot_over_line(turned, turned_dot, turned_line, alpha=None),
        _dot_over_line(straight, dot_at, line_at, alpha=None),
        rtol=1e-6,
    )


@pytest.mark.parametrize('ndim', [2, 3])
def test_meijering_default_alpha_is_the_documented_value(ndim):
    """`alpha=None` resolves to -1/(ndim+1) in every dimension."""
    rng = np.random.default_rng(0)
    image = ndi.gaussian_filter(rng.random((32,) * ndim), 1.5)

    def with_alpha(alpha):
        return meijering(image, sigmas=[2.0], alpha=alpha)

    sigmas = [2.0]
    meij_def_res = meijering(image, sigmas=sigmas)
    def_alpha = -1 / (ndim + 1)
    # Default alpha is -1 (ndim + 1)
    assert_array_equal(meij_def_res, meijering(image, sigmas=sigmas, alpha=def_alpha))
    # Reversing the sign of alpha gives a different result.
    assert not np.allclose(
        meij_def_res, meijering(image, sigmas=sigmas, alpha=-def_alpha)
    )
