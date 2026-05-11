import pytest

from itertools import combinations

import math

import numpy as np
from scipy import ndimage as ndi

from skimage import data
from skimage.registration import affine
from skimage._shared.testing import assert_array_equal, assert_array_almost_equal
from skimage.registration._affine import (
    _parameter_vector_to_matrix,
    _matrix_to_parameter_vector,
    _scale_matrix,
)

from skimage.registration import (
    solver_affine_lucas_kanade,
    solver_affine_studholme,
    solver_affine_ecc,
    target_registration_error,
)

solvers = [solver_affine_lucas_kanade, solver_affine_ecc, solver_affine_studholme]
models = ["affine", "euclidean", "translation"]
max_error = 2


# define the datasets used for the tests
@pytest.fixture
def data_2d_grayscale():
    return data.camera()[::8, ::8]


@pytest.fixture
def data_2d_rgb():
    return data.astronaut()[::8, ::8]


@pytest.fixture
def data_3d():
    return data.cells3d()[:, 1, ::8, ::8]


def create_matrix(shape, model, *, ndim):
    """
    Create a homogeneous test matrix

    Parameters
    ----------
    shape: tuple
        shape [D,W,W]
    model: str {'translation', 'euclidean' or 'affine'}
        Type of model
    ndim: int
        number of dimensions (2 or 3)

    Returns
    -------
    matrix: (ndim+1, ndim+1) ndarray
        The homogeneous transformation matrix
    """
    # Center the transformations
    T = np.eye(ndim + 1, dtype=np.float64)
    T[:ndim, -1] = np.array(shape) / 2

    matrix = np.eye(ndim + 1, dtype=np.float64)
    if model == "translation":
        matrix[:ndim, -1] += np.random.uniform(-2, 2, size=(ndim))
    elif model == "euclidean":
        R = np.eye(ndim + 1, dtype=np.float64)
        # Rotations for each planes
        for k, a in enumerate(combinations(range(ndim), 2)):
            r = np.random.uniform(-0.01, 0.01)
            c, s = math.cos(r), math.sin(r)
            R[a[0], a[0]] = c
            R[a[0], a[1]] = -s
            R[a[1], a[1]] = c
            R[a[1], a[0]] = s
            matrix = matrix @ R
        matrix[:ndim, -1] += np.random.uniform(-1, 1, size=(ndim))
    else:
        matrix[:ndim, :] += np.random.uniform(-0.01, 0.01, size=(ndim, ndim + 1))

    return np.linalg.inv(T) @ matrix @ T


def test_tre():
    tre = target_registration_error([10, 10], np.eye(3))
    assert_array_equal(np.zeros((10, 10)), tre)
    tre = target_registration_error(
        [10, 10], np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
    )
    assert_array_equal(np.ones((10, 10)), tre)


@pytest.mark.parametrize("solver", solvers)
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_register_affine_dtype(data_2d_grayscale, solver, dtype):
    reference = data_2d_grayscale
    forward = create_matrix(reference.shape, "affine", ndim=2)
    moving = ndi.affine_transform(reference, forward)
    matrix = affine(reference.astype(dtype), moving.astype(dtype), solver=solver)
    tre = target_registration_error(reference.shape, matrix @ forward)
    assert (
        tre.max() < max_error
    ), f"TRE ({tre.max():.2f}) is more than {max_error} pixels."


@pytest.mark.parametrize("solver", solvers)
@pytest.mark.parametrize("model", models)
def test_register_affine_model(data_2d_grayscale, solver, model):
    reference = data_2d_grayscale
    forward = create_matrix(reference.shape, model, ndim=2)
    moving = ndi.affine_transform(reference, forward)
    matrix = affine(reference, moving, solver=solver, model=model)
    tre = target_registration_error(reference.shape, matrix @ forward)
    assert (
        tre.max() < max_error
    ), f"TRE ({tre.max():.2f}) is more than {max_error} pixels."


@pytest.mark.parametrize("solver", solvers)
def test_register_affine_multichannel(data_2d_rgb, solver):
    reference = data_2d_rgb
    shp = [reference.shape[0], reference.shape[1]]
    forward = create_matrix(shp, "affine", ndim=2)
    moving = np.empty_like(reference)
    for ch in range(reference.shape[-1]):
        ndi.affine_transform(reference[..., ch], forward, output=moving[..., ch])
    matrix = affine(reference, moving, channel_axis=2, solver=solver)
    tre = target_registration_error(shp, matrix @ forward)
    assert (
        tre.max() < max_error
    ), f"TRE ({tre.max():.2f}) is more than {max_error} pixels."


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("ndim", [2, 3])
def test_matrix_parameter_vector_conversion(model, ndim):
    if model == "translation":
        p = np.zeros(ndim)
    elif model == "eucliean":
        p = np.zeros(ndim + len(combinations(range(ndim), 2)))
    else:
        p = np.zeros(ndim * (ndim + 1))
    m = _parameter_vector_to_matrix(p, model, ndim)
    assert m.shape == (ndim + 1, ndim + 1), f"Shape missmatch {m.shape}"


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("ndim", [2, 3])
def test_matrix_parameter_vector_conversion2(model, ndim):
    shape = (512, 512) if ndim == 2 else (512, 512, 512)
    matrix0 = create_matrix(shape, model, ndim=ndim)
    params = _matrix_to_parameter_vector(matrix0, model)
    matrix1 = _parameter_vector_to_matrix(params, model, ndim)
    assert_array_almost_equal(matrix0, matrix1, 1)


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("ndim", [2, 3])
def test_scale_matrix(model, ndim):
    shape = (512, 512) if ndim == 2 else (512, 512, 512)
    scale = 2.0
    matrix = create_matrix(shape, model, ndim=ndim)
    scaled = _scale_matrix(matrix, scale)
    assert_array_almost_equal(scale * matrix[:ndim, -1], scaled[:ndim, -1], 3)
    assert_array_almost_equal(matrix[:ndim, :ndim], scaled[:ndim, :ndim], 3)


@pytest.mark.parametrize("solver", solvers)
@pytest.mark.parametrize("model", models)
def test_inital_parameters(data_2d_grayscale, solver, model):
    ndim = 2
    reference = data_2d_grayscale
    forward = create_matrix(reference.shape, model, ndim=ndim)
    moving = ndi.affine_transform(reference, forward)
    initial_guess = np.linalg.inv(forward)
    matrix = affine(reference, moving, model=model, matrix=initial_guess, solver=solver)
    tre = target_registration_error(reference.shape, matrix @ forward)
    assert (
        tre.max() < max_error
    ), f"TRE ({tre.max():.2f}) is more than {max_error} pixels. {matrix.ravel()}"


@pytest.mark.parametrize("solver", solvers)
def test_3d(data_3d, solver):
    reference = data_3d
    forward = create_matrix(reference.shape, "affine", ndim=3)
    moving = ndi.affine_transform(reference, forward)
    matrix = affine(reference, moving, solver=solver)
    tre = target_registration_error(reference.shape, matrix @ forward)
    assert (
        tre.max() < max_error
    ), f"TRE ({tre.max():.2f}) is more than {max_error} pixels."


@pytest.mark.parametrize("solver", solvers)
def test_nomotion(data_2d_grayscale, solver):
    fixed = data_2d_grayscale
    matrix = affine(fixed, fixed, solver=solver)
    tre = target_registration_error(fixed.shape, matrix)
    assert (
        tre.max() < max_error
    ), f"TRE ({tre.max():.2f}) is more than {max_error} pixels."


@pytest.mark.parametrize("solver", solvers)
def test_weights(data_2d_grayscale, solver):
    reference = data_2d_grayscale
    forward = create_matrix(reference.shape, "translation", ndim=2)
    weights = ndi.median_filter(reference, 15) < 128
    moving = ndi.affine_transform(reference, forward)
    matrix = affine(
        reference, moving, weights=weights, model="translation", solver=solver
    )
    tre = target_registration_error(reference.shape, matrix @ forward)
    assert (
        tre.max() < max_error
    ), f"TRE ({tre.max():.2f}) is more than {max_error} pixels."
