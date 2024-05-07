import pytest
import numpy as np
from scipy import ndimage as ndi
from skimage.data import astronaut, cells3d
from skimage.registration import (
    affine,
    lucas_kanade_affine_solver,
    studholme_affine_solver,
)

solvers = [lucas_kanade_affine_solver, studholme_affine_solver]


def metrics(reference, moving, ground_truth_matrix, estimated_matrix):
    ndim = reference.ndim
    registered = ndi.affine_transform(moving, estimated_matrix)
    mask = registered > 0
    delta = reference - moving
    mse = np.mean(delta[mask] ** 2)
    if mse > 0:
        psnr = 20 * np.log(reference.max() / mse)
    else:
        psnr = np.inf

    matrix = np.linalg.inv(estimated_matrix)
    error1 = np.linalg.norm(matrix[:ndim, :ndim] - ground_truth_matrix[:ndim, :ndim])
    error2 = np.linalg.norm(matrix[:ndim, -1] - ground_truth_matrix[:ndim, -1])
    return psnr, error1, error2


@pytest.mark.parametrize("solver", solvers)
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_dtypes(dtype, solver):
    reference = astronaut()[..., 0]
    r = -0.12
    c, s = np.cos(r), np.sin(r)
    matrix_transform = np.array([[c, -s, 0], [s, c, 50], [0, 0, 1]])
    moving = ndi.affine_transform(reference, matrix_transform)
    matrix = affine(reference.astype(dtype), moving.astype(dtype), solver=solver)
    psnr, error1, error2 = metrics(reference, moving, matrix_transform, matrix)
    assert psnr > 20, f"PSNR({psnr}) < 20 "
    assert error1 < 0.1, f"Error norm({error1}) > 0.1 "
    assert error2 < 1, f"Error norm({error2}) > 1"


@pytest.mark.parametrize("solver", solvers)
def test_nomotion(solver):
    fixed = astronaut()[::4, ::4, 0]
    matrix = affine(fixed, fixed, solver=solver)
    psnr, error1, error2 = metrics(fixed, fixed, np.eye(3, 3), matrix)
    assert psnr > 20, f"PSNR({psnr}) < 20 "
    assert error1 < 0.1, f"Error norm({error1}) > 0.1 "
    assert error2 < 1, f"Error norm({error2}) > 1"


@pytest.mark.parametrize("solver", solvers)
def test_3d(solver):
    reference = cells3d()[:, 1, ::4, ::4]

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
    registered = ndi.affine_transform(moving, matrix)
    psnr, error1, error2 = metrics(reference, registered, matrix_transform, matrix)
    assert psnr > 10, f"PSNR({psnr}) < 10 "
    assert error1 < 0.1, f"Error norm({error1}) > 0.1 "
    assert error2 < 1, f"Error norm({error2}) > 1"
