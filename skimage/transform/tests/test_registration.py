import numpy as np
from scipy import ndimage as ndi

from skimage import data
from skimage.transform import registration
from skimage._shared.testing import (assert_array_equal,
                                     assert_array_almost_equal)


def test_register_affine():
    reference = data.camera()[::4, ::4]  # speed things up a little
    forward = np.array([[1.1, 0, 0],
                        [0  , 1, 0],
                        [0  , 0, 1]])

    inverse = np.linalg.inv(forward)

    target = ndi.affine_transform(reference, forward)
    matrix = registration.register_affine(reference, target)
    assert_array_almost_equal(matrix, inverse, decimal=1)


def test_register_affine_multichannel():
    reference = data.astronaut()[::4, ::4]  # speed things up a little
    forward = np.array([[1.1, 0, 0],
                        [0  , 1, 0],
                        [0  , 0, 1]])
    inverse = np.linalg.inv(forward)
    target = np.empty_like(reference)
    for ch in range(reference.shape[-1]):
        ndi.affine_transform(reference[..., ch], forward,
                             output=target[..., ch])
    matrix = registration.register_affine(reference, target,
                                          multichannel=True)
    assert_array_almost_equal(matrix, inverse, decimal=1)


def test_matrix_parameter_vector_conversion():
    for ndim in range(2, 5):
        p_v = np.random.rand((ndim + 1) * ndim)
        matrix = registration._parameter_vector_to_matrix(p_v, ndim)
        p_v_2 = registration._matrix_to_parameter_vector(matrix)
        assert_array_equal(p_v, p_v_2)
