import pytest
import numpy as np
from scipy import ndimage as ndi
from skimage.data import astronaut
from skimage.registration import parametric_ilk, parametric_nmi


def metrics(reference, moving, ground_truth_matrix, estimated_matrix):
    registered = ndi.affine_transform(moving, estimated_matrix)
    w = (registered > 0).astype(float)
    mse = (w * (registered - reference) ** 2).sum() / w.sum()
    matrix = np.linalg.inv(estimated_matrix)
    error1 = np.linalg.norm(matrix[:2, :2] - ground_truth_matrix[:2, :2])
    error2 = np.linalg.norm(matrix[2, -1] - ground_truth_matrix[2, -1])
    return mse, error1, error2


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_dtypes_ilk(dtype):
    reference = astronaut()[..., 0]
    r = -0.12
    c, s = np.cos(r), np.sin(r)
    matrix_transform = np.array([[c, -s, 0], [s, c, 50], [0, 0, 1]])
    moving = ndi.affine_transform(reference, matrix_transform)
    matrix = parametric_ilk(reference.astype(dtype), moving.astype(dtype), num_warp=20)
    mse, error1, error2 = metrics(reference, moving, matrix_transform, matrix)
    assert mse < 10, f"MSE({mse}) > 10 "
    assert error1 < 0.1, f"Error norm({error1}) > 0.1 "
    assert error2 < 1, f"Error norm({error2}) > 1"


def test_nomotion_ilk():
    fixed = astronaut()[::4, ::4, 0]
    matrix = parametric_ilk(fixed, fixed, num_warp=20)
    mse, error1, error2 = metrics(fixed, fixed, np.eye(3, 3), matrix)
    assert mse < 10, f"MSE({mse}) > 10 "
    assert error1 < 0.1, f"Error norm({error1}) > 0.1 "
    assert error2 < 1, f"Error norm({error2}) > 1"


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_dtypes_nmi(dtype):
    reference = astronaut()[..., 0]
    r = -0.12
    c, s = np.cos(r), np.sin(r)
    matrix_transform = np.array([[c, -s, 0], [s, c, 50], [0, 0, 1]])
    moving = ndi.affine_transform(reference, matrix_transform)
    matrix = parametric_nmi(reference.astype(dtype), moving.astype(dtype), num_warp=20)
    mse, error1, error2 = metrics(reference, moving, matrix_transform, matrix)
    assert mse < 10, f"MSE({mse}) > 10 "
    assert error1 < 0.1, f"Error norm({error1}) > 0.1 "
    assert error2 < 1, f"Error norm({error2}) > 1"


def test_nomotion_nmi():
    fixed = astronaut()[::4, ::4, 0]
    matrix = parametric_nmi(fixed, fixed)
    mse, error1, error2 = metrics(fixed, fixed, np.eye(3, 3), matrix)
    assert mse < 10, f"MSE({mse}) > 10 "
    assert error1 < 0.1, f"Error norm({error1}) > 0.1 "
    assert error2 < 1, f"Error norm({error2}) > 1"
