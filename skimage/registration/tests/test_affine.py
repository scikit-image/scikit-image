import pytest

from itertools import combinations

import math

import numpy as np
from scipy import ndimage as ndi

from skimage import data
from skimage.registration import affine
from skimage._shared.testing import assert_array_equal
from skimage.registration._affine import _parameter_vector_to_matrix, _scale_parameters

from skimage.registration import (
    lucas_kanade_affine_solver,
    studholme_affine_solver,
    target_registration_error,
)

solvers = [lucas_kanade_affine_solver, studholme_affine_solver]
models = ["affine", "euclidean", "translation"]
max_error = 2


def create_matrix(shape, model, ndim):
    """Create a homogeneous test matrix"""
    # Center the transformations
    T = np.eye(ndim + 1, dtype=np.float64)
    T[:ndim, -1] = np.array(shape) / 2

    matrix = np.eye(ndim + 1, dtype=np.float64)
    if model == "translation":
        matrix[:ndim, -1] += np.random.uniform(-10, 10, size=(ndim))
    elif model == "euclidean":
        R = np.eye(ndim + 1, dtype=np.float64)
        # Rotations for each planes
        for k, a in enumerate(combinations(range(ndim), 2)):
            r = np.random.uniform(-0.1, 0.1)
            c, s = math.cos(r), math.sin(r)
            R[a[0], a[0]] = c
            R[a[0], a[1]] = -s
            R[a[1], a[1]] = c
            R[a[1], a[0]] = s
            matrix @= R
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
def test_register_affine_dtype(solver, dtype):
    reference = data.camera()[::4, ::4]  # speed things up a little
    forward = create_matrix(reference.shape, "affine", 2)
    target = ndi.affine_transform(reference, forward)
    matrix = affine(reference.astype(dtype), target.astype(dtype), solver=solver)
    tre = target_registration_error(reference.shape, matrix @ forward)
    assert (
        tre.max() < max_error
    ), f"TRE ({tre.max():.2f}) is more than {max_error} pixels."


@pytest.mark.parametrize("solver", solvers)
@pytest.mark.parametrize("model", models)
def test_register_affine_model(solver, model):
    reference = data.camera()[::4, ::4].astype(float)  # speed things up a little
    forward = create_matrix(reference.shape, model, 2)
    target = ndi.affine_transform(reference, forward)
    matrix = affine(reference, target, solver=solver, model=model)
    tre = target_registration_error(reference.shape, matrix @ forward)
    assert (
        tre.max() < max_error
    ), f"TRE ({tre.max():.2f}) is more than {max_error} pixels."


@pytest.mark.parametrize("solver", solvers)
def test_register_affine_multichannel(solver):
    reference = data.astronaut()[::4, ::4]  # speed things up a little
    shp = [reference.shape[0], reference.shape[1]]
    forward = create_matrix(shp, "affine", 2)
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
    assert m.shape == (ndim + 1, ndim + 1), f"shape missmatch {m.shape}"


@pytest.mark.parametrize("ndim", [2, 3])
def test_scale_parameters(ndim):
    matrix = np.eye(ndim + 1)
    scaled = _scale_parameters(matrix, "affine", ndim, 2)
    assert_array_equal(2 * matrix[:ndim, -1], scaled[:ndim, -1])
    assert_array_equal(matrix[:ndim, :ndim], scaled[:ndim, :ndim])


@pytest.mark.parametrize("solver", solvers)
def test_3d(solver):
    reference = data.cells3d()[:, 1, ::4, ::4]
    forward = create_matrix(reference.shape, "affine", 3)
    moving = ndi.affine_transform(reference, forward)
    matrix = affine(reference, moving, solver=solver)
    tre = target_registration_error(reference.shape, matrix @ forward)
    assert (
        tre.max() < max_error
    ), f"TRE ({tre.max():.2f}) is more than {max_error} pixels."


@pytest.mark.parametrize("solver", solvers)
def test_nomotion(solver):
    fixed = data.camera()[::4, ::4]
    matrix = affine(fixed, fixed, solver=solver)
    tre = target_registration_error(fixed.shape, matrix)
    assert (
        tre.max() < max_error
    ), f"TRE ({tre.max():.2f}) is more than {max_error} pixels."


@pytest.mark.parametrize("solver", solvers)
def test_weights(solver):
    reference = data.camera()[::4, ::4]
    forward = create_matrix(reference.shape, "translation", 2)
    weights = ndi.median_filter(reference, 15) < 128
    moving = ndi.affine_transform(reference, forward)
    matrix = affine(reference, moving, weights=weights, model="translation")
    tre = target_registration_error(reference.shape, matrix @ forward)
    assert (
        tre.max() < max_error
    ), f"TRE ({tre.max():.2f}) is more than {max_error} pixels."
