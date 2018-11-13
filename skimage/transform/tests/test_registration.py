import numpy as np
from scipy import ndimage as ndi

from skimage import data
from skimage.transform import registration
from skimage._shared import testing
from skimage._shared.testing import (assert_array_equal,
                                     assert_array_almost_equal)
from skimage._shared._warnings import expected_warnings


def test_register_affine():
    reference = data.camera()
    inner_matrix = [[1.1, 0],
                    [0, 1]]
    right_matrix = np.zeros((2, 1))
    bottom_matrix = [[0, 0, 1]]
    matrix_transform = np.concatenate((inner_matrix, right_matrix), axis=1)
    matrix_transform = np.concatenate(
        (matrix_transform, bottom_matrix), axis=0)

    inverse_inner = np.linalg.inv(inner_matrix)
    inverse_right = -np.matmul(inverse_inner, right_matrix)
    inverse_transform = np.concatenate((inverse_inner, inverse_right), axis=1)
    inverse_transform = np.concatenate(
        (inverse_transform, bottom_matrix), axis=0)

    target = ndi.affine_transform(reference, matrix_transform)
    with expected_warnings(['The default multichannel']):
        matrix = registration.register_affine(reference, target)
        assert_array_almost_equal(matrix, inverse_transform, decimal=0)


def test_matrix_parameter_vector_conversion():
    for ndim in range(2, 5):
        p_v = np.random.rand((ndim + 1) * ndim)
        matrix = registration._parameter_vector_to_matrix(p_v, ndim)
        p_v_2 = registration._matrix_to_parameter_vector(matrix)
        assert_array_equal(p_v, p_v_2)
