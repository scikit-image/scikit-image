import operator
from functools import partial

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_compare,
    assert_array_less,
    assert_equal,
)

from skimage import img_as_float
from _skimage2._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.data import camera, retina
from skimage.filters import frangi, hessian, jerman, meijering, sato
from skimage.util import crop, invert


def test_2d_null_matrix():
    a_black = np.zeros((3, 3)).astype(np.uint8)
    a_white = invert(a_black)

    zeros = np.zeros((3, 3))
    ones = np.ones((3, 3))

    assert_equal(meijering(a_black, black_ridges=True), zeros)
    assert_equal(meijering(a_white, black_ridges=False), zeros)

    assert_equal(sato(a_black, black_ridges=True, mode='reflect'), zeros)
    assert_equal(sato(a_white, black_ridges=False, mode='reflect'), zeros)

    assert_equal(jerman(a_black, black_ridges=True, mode='reflect'), zeros)
    assert_equal(jerman(a_white, black_ridges=False, mode='reflect'), zeros)

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

    assert_equal(jerman(a_black, black_ridges=True, mode='reflect'), zeros)
    assert_equal(jerman(a_white, black_ridges=False, mode='reflect'), zeros)

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

    assert_array_less(
        jerman(a_black, black_ridges=True, mode='reflect').std(), a_black.std()
    )
    assert_array_less(
        jerman(a_white, black_ridges=False, mode='reflect').std(), a_white.std()
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

    assert_array_less(
        jerman(a_black, black_ridges=True, mode='reflect').std(), a_black.std()
    )
    assert_array_less(
        jerman(a_white, black_ridges=False, mode='reflect').std(), a_white.std()
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
        jerman(1 * a_black, black_ridges=True, mode='reflect'),
        jerman(10 * a_black, black_ridges=True, mode='reflect'),
        atol=1e-3,
    )
    assert_allclose(
        jerman(1 * a_white, black_ridges=False, mode='reflect'),
        jerman(10 * a_white, black_ridges=False, mode='reflect'),
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
        jerman(1 * a_black, black_ridges=True, mode='reflect'),
        jerman(10 * a_black, black_ridges=True, mode='reflect'),
        atol=1e-3,
    )
    assert_allclose(
        jerman(1 * a_white, black_ridges=False, mode='reflect'),
        jerman(10 * a_white, black_ridges=False, mode='reflect'),
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
        jerman(a_black, black_ridges=True, mode='reflect'),
        jerman(a_white, black_ridges=False, mode='reflect'),
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


@pytest.mark.parametrize('func', [meijering, sato, jerman, frangi, hessian])
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
        jerman(a_black, black_ridges=True, mode='reflect'),
        jerman(a_white, black_ridges=False, mode='reflect'),
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
    'func, tol',
    [(frangi, 1e-2), (meijering, 1e-2), (sato, 2e-3), (jerman, 2e-2), (hessian, 2e-2)],
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


def test_jerman_result_decrease_with_tau_increase():
    # tau is a parameter of the Jerman filter used to enhance the difference between
    # lamdba3 and lambda2. Increasing tau makes the filter less sensitive to noisy
    # structures.
    #
    # The Jerman filter depends only on the ratio of lambda_rho / lambda2, where
    # lambda_rho is the enhanced lambda3 (lambda3 can only increase with tau, see
    # implementation). The implementation also ensures lambda_rho >= lambda2, so r =
    # lambda_rho / lambda2 >= 1. The final value of the filter depends on:
    #
    # V = lambda2**2 * (lambda_rho - lambda2) * 27 / ((lambda2 + lambda_rho) ** 3) with
    # V = 1 if lambda2 >= lambda_rho/2 > 0
    #
    # Substituting lambda_rho = lambda2 * r:
    #
    # V = 1 if 2 >= r > 1, otherwise for r > 2:
    #
    # V(r) = lambda2**2 * (lambda2 * (r - 1)) * 27 / ((lambda2 * (1 + r)) ** 3)
    # V(r) = (r - 1) * 27 / ((1 + r) ** 3)
    #
    # We can verify that V(r) decreses with r > 2 with the derivative of V that is
    # strictly negative for r > 2:
    #
    # V'(r) = 27 * (4 - 2*r) / ((1 + r) ** 4)
    #
    # r increases with tau (or stays the same). V decreases with r (or stays the same).
    assert_array_less_or_equal = partial(assert_array_compare, operator.le)

    img = rgb2gray(retina()[300:500, 700:900])
    monotonic_tau_increase = np.linspace(0, 2, 5)

    out = jerman(img, tau=monotonic_tau_increase[0])
    for tau in monotonic_tau_increase[1:]:
        out_next = jerman(img, tau=tau)
        assert_array_less_or_equal(out_next, out)
        # for the retina image, we know we do not max out the filter response, so we
        # expect a decrease in the sum of all values.
        assert out_next.sum() <= out.sum()
        out = out_next
