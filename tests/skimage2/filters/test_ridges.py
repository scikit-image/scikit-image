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
# smoothing at which we test.  This is because blurring costs a round dot more
# height than a long line.  Scaling the dot by ``sqrt(1 + sigma**2/width**2)``
# cancels that, so dot and line tie exactly at ``alpha = 0`` and any later
# inequality is result of the alpha parameter.

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
    assert_allclose(_dot_line_ratio(image, dot_at, line_at, alpha=0.0),
                    1.0,
                    rtol=1e-6)
    # Ratio should be 2 / 3 for default alpha, preferring line to dot.
    def_dot_line = _dot_line_ratio(image, dot_at, line_at, alpha=None)
    assert np.isclose(def_dot_line , expected_ratio, rtol=1e-6)
    # Check that alpha=None corresponds to stated default.
    assert_equal(_dot_line_ratio(image, dot_at, line_at, alpha=def_alpha),
                 def_dot_line)
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
    assert_array_equal(rand_def_dot_line,
                       meijering(rand_image, sigmas=sigmas, alpha=def_alpha))
    # Reversing the sign of alpha gives a different result.
    assert not np.allclose(
        rand_def_dot_line,
        meijering(rand_image, sigmas=sigmas, alpha=-def_alpha)
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
    straight_ratio = _dot_line_ratio(straight, dot_at, line_at, alpha=None),
    for angle in [15, 30, 45, 63, 90]:
        turned, turned_dot, turned_line = _dot_and_line(angle=angle)
        assert_allclose(
            _dot_line_ratio(turned, turned_dot, turned_line, alpha=None),
            straight_ratio,
            rtol=1e-6,
        )
