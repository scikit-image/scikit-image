import pytest

import numpy as np
from scipy import ndimage as ndi

from skimage import data
from skimage._shared.testing import assert_array_equal, assert_array_almost_equal
from skimage.registration import affine
from skimage.registration._affine import (
    _parameter_vector_to_matrix,
    _matrix_to_parameter_vector,
)
from skimage.registration import (
    lucas_kanade_affine_solver,
    studholme_affine_solver,
)

solvers = [lucas_kanade_affine_solver, studholme_affine_solver]


@pytest.mark.parametrize("solver", solvers)
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_register_affine(solver, dtype):
    reference = data.camera()[::4, ::4]  # speed things up a little
    forward = np.array([[1.1, 0, 0], [0, 1, 0], [0, 0, 1]])
    inverse = np.linalg.inv(forward)
    target = ndi.affine_transform(reference, forward)
    matrix = affine(reference.astype(dtype), target.astype(dtype), solver=solver)
    # distinguish errors on the 2x2 sub matrix and translation
    assert_array_almost_equal(matrix[:2, :2], inverse[:2, :2], decimal=1)
    assert_array_almost_equal(matrix[:2, -1], inverse[:2, -1], decimal=0)


@pytest.mark.parametrize("solver", solvers)
def test_register_affine_multichannel(solver):
    # reference = data.astronaut()[::4, ::4]  # speed things up a little
    # forward = np.array([[1.01, 0, 0], [0, 1, 0], [0, 0, 1]])
    # inverse = np.linalg.inv(forward)
    # target = np.empty_like(reference)
    # for ch in range(reference.shape[-1]):
    #     ndi.affine_transform(reference[..., ch], forward, output=target[..., ch])
    # matrix = affine(reference, target, channel_axis=2)
    reference = data.astronaut()[::4, ::4]  # speed things up a little
    forward = np.array([[1, 0, 0], [0, 1.1, 0], [0, 0, 1]])
    inverse = np.linalg.inv(forward)
    moving = np.empty_like(reference)
    # moving = ndi.affine_transform(reference, forward)
    for ch in range(reference.shape[-1]):
        ndi.affine_transform(reference[..., ch], forward, output=moving[..., ch])
    matrix = affine(reference, moving, channel_axis=2)

    # distinguish errors on the 2x2 sub matrix and translation
    assert_array_almost_equal(matrix[:2, :2], inverse[:2, :2], decimal=1)
    assert_array_almost_equal(matrix[:2, -1], inverse[:2, -1], decimal=0)


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_matrix_parameter_vector_conversion(ndim):
    p = np.random.rand((ndim + 1) * ndim)
    m = _parameter_vector_to_matrix(p, ndim)
    p2 = _matrix_to_parameter_vector(m, ndim)
    assert_array_equal(p, p2)


@pytest.mark.parametrize("solver", solvers)
def test_3d(solver):
    reference = data.cells3d()[:, 1, ::4, ::4]

    T = np.concatenate(
        [
            np.concatenate(
                [np.eye(3), -np.array(reference.shape).reshape(3, 1) / 2], axis=1
            ),
            [[0, 0, 0, 1]],
        ]
    )
    r1 = np.random.uniform(-0.2, 0.2)  # radians
    c1, s1 = np.cos(r1), np.sin(r1)
    R1 = np.array([[c1, -s1, 0, 0], [s1, c1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    r2 = np.random.uniform(-0.2, 0.2)  # radians
    c2, s2 = np.cos(r2), np.sin(r2)
    R2 = np.array([[c2, -s2, 0, 0], [s2, c2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    matrix_transform = np.linalg.inv(T) @ R2 @ R1 @ T
    moving = ndi.affine_transform(reference, matrix_transform)
    matrix = affine(reference, moving, solver=solver)
    inverse = np.linalg.inv(matrix_transform)
    # registered = ndi.affine_transform(moving, matrix)
    assert_array_almost_equal(matrix[:2, :2], inverse[:2, :2], decimal=1)
    assert_array_almost_equal(matrix[:2, -1], inverse[:2, -1], decimal=0)


@pytest.mark.parametrize("solver", solvers)
def test_nomotion(solver):
    fixed = data.astronaut()[::4, ::4, 0]
    matrix = affine(fixed, fixed, solver=solver)
    assert_array_almost_equal(matrix[:2, :2], np.eye(2), decimal=1)
    assert_array_almost_equal(matrix[:2, -1], np.zeros(2), decimal=0)
