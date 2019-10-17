
import numpy as np
from numpy.testing import assert_allclose, assert_array_less, assert_equal
from skimage.filters import meijering, sato, frangi, hessian
from skimage.data import camera
from skimage.util import crop, invert


def test_2d_null_matrix():

    a_black = np.zeros((3, 3)).astype(np.uint8)
    a_white = invert(a_black)

    zeros = np.zeros((3, 3))
    ones = np.ones((3, 3))

    assert_equal(meijering(a_black, black_ridges=True), ones)
    assert_equal(meijering(a_white, black_ridges=False), ones)

    assert_equal(sato(a_black, black_ridges=True), zeros)
    assert_equal(sato(a_white, black_ridges=False), zeros)

    assert_allclose(frangi(a_black, black_ridges=True), zeros, atol=1e-3)
    assert_allclose(frangi(a_white, black_ridges=False), zeros, atol=1e-3)

    assert_equal(hessian(a_black, black_ridges=False), ones)
    assert_equal(hessian(a_white, black_ridges=True), ones)


def test_3d_null_matrix():

    a_black = np.zeros((3, 3, 3)).astype(np.uint8)
    a_white = invert(a_black)

    zeros = np.zeros((3, 3, 3))
    ones = np.ones((3, 3, 3))

    assert_allclose(meijering(a_black, black_ridges=True), ones, atol=1e-1)
    assert_allclose(meijering(a_white, black_ridges=False), ones, atol=1e-1)

    assert_equal(sato(a_black, black_ridges=True), zeros)
    assert_equal(sato(a_white, black_ridges=False), zeros)

    assert_allclose(frangi(a_black, black_ridges=True), zeros, atol=1e-3)
    assert_allclose(frangi(a_white, black_ridges=False), zeros, atol=1e-3)

    assert_equal(hessian(a_black, black_ridges=False), ones)
    assert_equal(hessian(a_white, black_ridges=True), ones)


def test_2d_energy_decrease():

    a_black = np.zeros((5, 5)).astype(np.uint8)
    a_black[2, 2] = 255
    a_white = invert(a_black)

    assert_array_less(meijering(a_black, black_ridges=True).std(),
                      a_black.std())
    assert_array_less(meijering(a_white, black_ridges=False).std(),
                      a_white.std())

    assert_array_less(sato(a_black, black_ridges=True).std(), a_black.std())
    assert_array_less(sato(a_white, black_ridges=False).std(), a_white.std())

    assert_array_less(frangi(a_black, black_ridges=True).std(),
                      a_black.std())
    assert_array_less(frangi(a_white, black_ridges=False).std(),
                      a_white.std())

    assert_array_less(hessian(a_black, black_ridges=True).std(),
                      a_black.std())
    assert_array_less(hessian(a_white, black_ridges=False).std(),
                      a_white.std())


def test_3d_energy_decrease():

    a_black = np.zeros((5, 5, 5)).astype(np.uint8)
    a_black[2, 2, 2] = 255
    a_white = invert(a_black)

    assert_array_less(meijering(a_black, black_ridges=True).std(),
                      a_black.std())
    assert_array_less(meijering(a_white, black_ridges=False).std(),
                      a_white.std())

    assert_array_less(sato(a_black, black_ridges=True).std(), a_black.std())
    assert_array_less(sato(a_white, black_ridges=False).std(), a_white.std())

    assert_array_less(frangi(a_black, black_ridges=True).std(),
                      a_black.std())
    assert_array_less(frangi(a_white, black_ridges=False).std(),
                      a_white.std())

    assert_array_less(hessian(a_black, black_ridges=True).std(),
                      a_black.std())
    assert_array_less(hessian(a_white, black_ridges=False).std(),
                      a_white.std())


def test_2d_linearity():

    a_black = np.ones((3, 3)).astype(np.uint8)
    a_white = invert(a_black)

    assert_allclose(meijering(1 * a_black, black_ridges=True),
                    meijering(10 * a_black, black_ridges=True), atol=1e-3)
    assert_allclose(meijering(1 * a_white, black_ridges=False),
                    meijering(10 * a_white, black_ridges=False), atol=1e-3)

    assert_allclose(sato(1 * a_black, black_ridges=True),
                    sato(10 * a_black, black_ridges=True), atol=1e-3)
    assert_allclose(sato(1 * a_white, black_ridges=False),
                    sato(10 * a_white, black_ridges=False), atol=1e-3)

    assert_allclose(frangi(1 * a_black, black_ridges=True),
                    frangi(10 * a_black, black_ridges=True), atol=1e-3)
    assert_allclose(frangi(1 * a_white, black_ridges=False),
                    frangi(10 * a_white, black_ridges=False), atol=1e-3)

    assert_allclose(hessian(1 * a_black, black_ridges=True),
                    hessian(10 * a_black, black_ridges=True), atol=1e-3)
    assert_allclose(hessian(1 * a_white, black_ridges=False),
                    hessian(10 * a_white, black_ridges=False), atol=1e-3)


def test_3d_linearity():

    a_black = np.ones((3, 3, 3)).astype(np.uint8)
    a_white = invert(a_black)

    assert_allclose(meijering(1 * a_black, black_ridges=True),
                    meijering(10 * a_black, black_ridges=True), atol=1e-3)
    assert_allclose(meijering(1 * a_white, black_ridges=False),
                    meijering(10 * a_white, black_ridges=False), atol=1e-3)

    assert_allclose(sato(1 * a_black, black_ridges=True),
                    sato(10 * a_black, black_ridges=True), atol=1e-3)
    assert_allclose(sato(1 * a_white, black_ridges=False),
                    sato(10 * a_white, black_ridges=False), atol=1e-3)

    assert_allclose(frangi(1 * a_black, black_ridges=True),
                    frangi(10 * a_black, black_ridges=True), atol=1e-3)
    assert_allclose(frangi(1 * a_white, black_ridges=False),
                    frangi(10 * a_white, black_ridges=False), atol=1e-3)

    assert_allclose(hessian(1 * a_black, black_ridges=True),
                    hessian(10 * a_black, black_ridges=True), atol=1e-3)
    assert_allclose(hessian(1 * a_white, black_ridges=False),
                    hessian(10 * a_white, black_ridges=False), atol=1e-3)


def test_2d_cropped_camera_image():

    a_black = crop(camera(), ((206, 206), (206, 206)))
    a_white = invert(a_black)

    zeros = np.zeros((100, 100))
    ones = np.ones((100, 100))

    assert_allclose(meijering(a_black, black_ridges=True),
                    meijering(a_white, black_ridges=False))

    assert_allclose(sato(a_black, black_ridges=True),
                    sato(a_white, black_ridges=False))

    assert_allclose(frangi(a_black, black_ridges=True), zeros, atol=1e-3)
    assert_allclose(frangi(a_white, black_ridges=False), zeros, atol=1e-3)

    assert_allclose(hessian(a_black, black_ridges=True), ones, atol=1 - 1e-7)
    assert_allclose(hessian(a_white, black_ridges=False), ones, atol=1 - 1e-7)


def test_3d_cropped_camera_image():

    a_black = crop(camera(), ((206, 206), (206, 206)))
    a_black = np.dstack([a_black, a_black, a_black])
    a_white = invert(a_black)

    zeros = np.zeros((100, 100, 3))
    ones = np.ones((100, 100, 3))

    assert_allclose(meijering(a_black, black_ridges=True),
                    meijering(a_white, black_ridges=False))

    assert_allclose(sato(a_black, black_ridges=True),
                    sato(a_white, black_ridges=False))

    assert_allclose(frangi(a_black, black_ridges=True), zeros, atol=1e-3)
    assert_allclose(frangi(a_white, black_ridges=False), zeros, atol=1e-3)

    assert_allclose(hessian(a_black, black_ridges=True), ones, atol=1 - 1e-7)
    assert_allclose(hessian(a_white, black_ridges=False), ones, atol=1 - 1e-7)


if __name__ == "__main__":
    from numpy import testing
    testing.run_module_suite()
