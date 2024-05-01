import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy import ndimage
from skimage.data import astronaut
from skimage.registration import estimate_global_affine_motion


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_basic(dtype):
    moving = astronaut()[:, :, 0].astype(dtype)
    A0 = np.eye(2) + 0.01 * np.random.randn(2, 2)
    v0 = np.random.randn(2).flatten()
    fixed = ndimage.affine_transform(moving.astype(float), A0, v0)
    A1, v1 = estimate_global_affine_motion(fixed.astype(float), moving.astype(float))
    v1 = v1.flatten()
    assert_allclose(A0, A1, rtol=1e-3, atol=1e-3)
    assert_allclose(v0, v1, rtol=1e-3, atol=1e-3)


def test_nomotion():
    fixed = astronaut()[:, :, 0]
    A, v = estimate_global_affine_motion(fixed.astype(float), fixed.astype(float))
    assert_allclose(A, np.eye(2), rtol=1e-3, atol=1e-3)
    assert_allclose(v, np.zeros((2, 1)), rtol=1e-1, atol=1e-1)
