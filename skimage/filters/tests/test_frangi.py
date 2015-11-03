import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_allclose
from skimage.filters._frangi import frangi, hessian
from skimage.data import camera
from skimage.util import crop


def test_null_matrix():
    a = np.zeros((3, 3))
    assert_almost_equal(frangi(a), np.zeros((3, 3)))
    assert_almost_equal(frangi(a, black_ridges=False), np.zeros((3, 3)))
    assert_equal(hessian(a), np.ones((3, 3)))


def test_energy_decrease():
    a = np.zeros((3, 3))
    a[1, 1] = 1.
    assert frangi(a).std() < a.std()
    assert frangi(a, black_ridges=False).std() < a.std()
    assert hessian(a).std() > a.std()


def test_values_decreased():
    a = np.multiply(np.ones((3, 3)), 10)
    assert_equal(frangi(a), np.zeros((3, 3)))
    assert_equal(frangi(a, black_ridges=False), np.zeros((3, 3)))
    assert_equal(hessian(a), np.ones((3, 3)))


def test_cropped_camera_image():
    image = crop(camera(), ((206, 206), (206, 206)))
    assert_allclose(frangi(image), np.zeros((100, 100)), atol=1e-03)
    assert_allclose(frangi(image, black_ridges=True), np.zeros((100,100)), atol=1e-03)
    assert_allclose(hessian(image), np.ones((100, 100)), atol=1-1e-07)

if __name__ == "__main__":
    from numpy import testing
    testing.run_module_suite()
