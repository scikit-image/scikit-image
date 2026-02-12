import pytest

from itertools import combinations

import math

import numpy as np
from scipy import ndimage as ndi
import skimage as ski

max_error = 10

solvers = [
    ski.registration.StudholmeAffineSolver,
    ski.registration.LucasKanadeAffineSolver,
    ski.registration.ECCAffineSolver,
]

models = [
    ski.registration.TranslationTransform,
    ski.registration.EuclideanTransform,
    ski.registration.AffineTransform,
]


# define the datasets used for the tests
@pytest.fixture
def data_2d_grayscale():
    return ski.data.camera()[::8, ::8]


@pytest.fixture
def data_2d_rgb():
    return ski.data.astronaut()[::8, ::8]


@pytest.fixture
def data_3d():
    return ski.data.cells3d()[:, 1, ::8, ::8]


def create_matrix(shape, model):
    """Create a homogeneous test matrix

    Parameters
    ----------
    shape:
        Shape of the reference image.
    model: {translation,euclidean,affine}
        Model of affine deformation

    Returns
    -------
    transform: ndarray
        Affine transformation matrix
    """
    ndim = len(shape)
    # Center the transformations
    T = np.eye(ndim + 1, dtype=np.float64)
    T[:ndim, -1] = -np.array(shape) / 2
    matrix = np.eye(ndim + 1, dtype=np.float64)
    # translation
    matrix[:ndim, -1] += np.random.uniform(-2, 2, size=(ndim))
    if (
        model == ski.registration.EuclideanTransform
        or model == ski.registration.AffineTransform
    ):
        # Rotations for each planes
        for a in combinations(range(ndim), 2):
            R = np.eye(ndim + 1, dtype=np.float64)
            r = np.random.uniform(-np.pi / 10, np.pi / 10)
            c, s = math.cos(r), math.sin(r)
            R[a[0], a[0]] = c
            R[a[1], a[0]] = -s
            R[a[0], a[1]] = s
            R[a[1], a[1]] = c
            matrix = matrix @ R

    if model == ski.registration.AffineTransform:
        # Shear for each plane
        for a in combinations(range(ndim), 2):
            r = np.random.uniform(-0.01, 0.01)
            S = np.eye(ndim + 1)
            S[a[0], a[1]] = r
            matrix = S @ matrix
        # Zoom
        Z = np.eye(ndim + 1)
        for k in range(ndim):
            Z[k, k] = np.random.uniform(0.9, 1.1)
        matrix = Z @ matrix

    return np.linalg.inv(T) @ matrix @ T


def test_tre():
    tre = ski.registration.target_registration_error([10, 10], np.eye(3))
    np.testing.assert_allclose(np.zeros((10, 10)), tre, 1e-3)
    tre = ski.registration.target_registration_error(
        [10, 10], np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
    )
    np.testing.assert_allclose(np.ones((10, 10)), tre, 1e-3)


def test_shuffle():
    from skimage.registration._affine import (
        _shuffle_axes_and_unpack_weights_if_necessary,
    )

    data = _shuffle_axes_and_unpack_weights_if_necessary(np.zeros((10, 10)), None)
    assert len(data) == 2
    assert data[0].shape == (1, 10, 10)
    assert data[1].shape == (1, 10, 10)

    data = _shuffle_axes_and_unpack_weights_if_necessary(np.zeros((1, 10, 10)), 0)
    assert len(data) == 2
    assert data[0].shape == (1, 10, 10)
    assert data[1].shape == (1, 10, 10)

    data = _shuffle_axes_and_unpack_weights_if_necessary(
        (np.zeros((1, 10, 10)), np.zeros((1, 10, 10))), 0
    )
    assert len(data) == 2
    assert data[0].shape == (1, 10, 10)
    assert data[1].shape == (1, 10, 10)


def test_gaussian_pyramid():
    shape = (1, 32, 32)
    image = np.zeros(shape)
    g = ski.registration.GaussianPyramid(min_size=8)
    assert g.max_layers(shape) == 2  # size 16, 8
    p = g.generate(image, channel_axis=0)
    assert len(p) == 2  # image and weights
    assert len(p[0]) == 3  # 3 scales 8, 16, 32
    # test the shape of the elements of the pyramid
    for k in range(len(p[0])):
        n = int(shape[1] / np.pow(2.0, len(p[0]) - k - 1))
        assert p[0][k].shape == (1, n, n)
        assert p[1][k].shape == (1, n, n)


def test_no_pyramid():
    from skimage.registration._affine import _NullPyramid

    shape = (32, 32)
    image = np.zeros(shape)
    n = _NullPyramid()
    assert n.max_layers(shape) == 1
    p = n.generate(image, channel_axis=None)
    assert len(p) == 2
    assert len(p[0]) == 1


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("solver", solvers)
@pytest.mark.parametrize("model", models)
def test_matrix_to_parameters(ndim, solver, model):
    shape = (512, 512) if ndim == 2 else (512, 512, 512)
    matrix0 = create_matrix(shape, model)
    assert matrix0.shape == (len(shape) + 1, len(shape) + 1)
    s = solver(model)
    assert s._model_class == model
    params = s._matrix_to_parameter_vector(
        matrix0,
    )
    matrix1 = s._parameter_vector_to_matrix(params, ndim)
    assert matrix1.shape == (len(shape) + 1, len(shape) + 1)
    max_abs_error = np.max(np.abs(matrix0 - matrix1))
    assert max_abs_error < 0.001


@pytest.mark.parametrize("solver", solvers)
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_register_affine_dtype(data_2d_grayscale, solver, dtype):
    reference = data_2d_grayscale
    forward = create_matrix(reference.shape, ski.registration.TranslationTransform)
    moving = ndi.affine_transform(reference, forward)
    tfm = ski.registration.estimate_affine(
        reference.astype(dtype),
        moving.astype(dtype),
        solver=solver(ski.registration.TranslationTransform),
    )
    assert tfm.params.shape == (3, 3)
    tre = ski.registration.target_registration_error(
        reference.shape, tfm.params @ forward
    )
    assert (
        tre.max() < max_error
    ), f"TRE ({tre.max():.2f}) is more than {max_error} pixels."


@pytest.mark.parametrize("solver", solvers)
@pytest.mark.parametrize("model", models)
def test_register_affine_model(data_2d_grayscale, solver, model):
    reference = data_2d_grayscale
    forward = create_matrix(reference.shape, model)
    moving = ndi.affine_transform(reference, forward)
    tfm = ski.registration.estimate_affine(reference, moving, solver=solver(model))
    assert tfm.params.shape == (3, 3)
    tre = ski.registration.target_registration_error(
        reference.shape, tfm.params @ forward
    )
    tre_max = tre.max()
    assert tre_max < max_error, f"TRE ({tre_max:.2f}) is more than {max_error} pixels."


@pytest.mark.parametrize("solver", solvers)
@pytest.mark.parametrize("model", models)
def test_register_affine_init(data_2d_grayscale, solver, model):
    reference = data_2d_grayscale
    forward = create_matrix(reference.shape, model)
    backward = np.linalg.inv(forward)
    moving = ndi.affine_transform(reference, forward)
    tfm = ski.registration.estimate_affine(
        reference, moving, solver=solver(model), matrix=backward
    )
    assert tfm.params.shape == (3, 3)
    tre = ski.registration.target_registration_error(
        reference.shape, tfm.params @ forward
    )
    tre_max = tre.max()
    assert tre_max < max_error, f"TRE ({tre_max:.2f}) is more than {max_error} pixels."


@pytest.mark.parametrize("solver", solvers)
def test_register_affine_multichannel(data_2d_rgb, solver):
    reference = data_2d_rgb
    model = ski.registration.TranslationTransform
    shape = [reference.shape[0], reference.shape[1]]
    # forward = create_matrix(shape, model)
    forward = np.array([[1, 0, 4], [0, 1, 0], [0, 0, 1]])
    moving = np.empty_like(reference)
    for ch in range(reference.shape[-1]):
        ndi.affine_transform(reference[..., ch], forward, output=moving[..., ch])
    tfm = ski.registration.estimate_affine(
        reference, moving, solver=solver(model), channel_axis=2
    )
    tre = ski.registration.target_registration_error(shape, tfm.params @ forward)
    tre_max = tre.max()
    assert (
        tre_max < max_error
    ), f"TRE ({tre.max():.2f}) is more than {max_error} pixels."


@pytest.mark.parametrize("solver", solvers)
def test_3d(data_3d, solver):
    reference = data_3d
    model = ski.registration.TranslationTransform
    forward = np.array([[1, 0, 0, 4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    moving = ndi.affine_transform(reference, forward)
    tfm = ski.registration.estimate_affine(reference, moving, solver=solver(model))
    tre = ski.registration.target_registration_error(
        reference.shape, tfm.params @ forward
    )
    tre_max = tre.max()
    assert (
        tre_max < max_error
    ), f"TRE ({tre.max():.2f}) is more than {max_error} pixels."


@pytest.mark.parametrize("solver", solvers)
def test_register_weight(data_2d_grayscale, solver):
    model = ski.registration.TranslationTransform
    reference = data_2d_grayscale
    weight_reference = reference > 128
    forward = create_matrix(reference.shape, model)
    moving = ndi.affine_transform(reference, forward)
    weight_moving = moving > 128
    tfm = ski.registration.estimate_affine(
        (reference, weight_reference), (moving, weight_moving), solver=solver(model)
    )
    assert tfm.params.shape == (3, 3)
    tre = ski.registration.target_registration_error(
        reference.shape, tfm.params @ forward
    )
    tre_max = tre.max()
    assert tre_max < max_error, f"TRE ({tre_max:.2f}) is more than {max_error} pixels."


########################################

# @pytest.mark.parametrize("solver", solvers)
# def test_nomotion(data_2d_grayscale, solver):
#     fixed = data_2d_grayscale
#     matrix = affine(fixed, fixed, solver=solver)
#     tre = target_registration_error(fixed.shape, matrix)
#     assert tre.max() < max_error, (
#         f"TRE ({tre.max():.2f}) is more than {max_error} pixels."
#     )


# @pytest.mark.parametrize("solver", solvers)
# def test_weights(data_2d_grayscale, solver):
#     reference = data_2d_grayscale
#     forward = create_matrix(reference.shape, "translation", ndim=2)
#     weights = ndi.median_filter(reference, 15) < 128
#     moving = ndi.affine_transform(reference, forward)
#     matrix = affine(
#         reference, moving, weights=weights, model="translation", solver=solver
#     )
#     tre = target_registration_error(reference.shape, matrix @ forward)
#     assert tre.max() < max_error, (
#         f"TRE ({tre.max():.2f}) is more than {max_error} pixels."
#     )
