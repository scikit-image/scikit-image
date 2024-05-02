import numpy as np
from scipy import ndimage as ndi

from skimage import data
from skimage.registration import affine
from skimage.registration._affine import _parameter_vector_to_matrix
from skimage._shared.testing import assert_array_equal, assert_array_almost_equal


def test_register_affine():
    reference = data.camera()[::4, ::4]  # speed things up a little
    forward = np.array([[1.1, 0, 0], [0, 1, 0], [0, 0, 1]])

    inverse = np.linalg.inv(forward)

    target = ndi.affine_transform(reference, forward)
    matrix = affine(reference, target)
    # distinguish errors on the 2x2 sub matrix and translation
    assert_array_almost_equal(matrix[:2, :2], inverse[:2, :2], decimal=1)
    assert_array_almost_equal(matrix[:2, -1], inverse[:2, -1], decimal=0)


def test_register_affine_multichannel():
    reference = data.astronaut()[::4, ::4]  # speed things up a little
    forward = np.array([[1.1, 0, 0], [0, 1, 0], [0, 0, 1]])
    inverse = np.linalg.inv(forward)
    target = np.empty_like(reference)
    for ch in range(reference.shape[-1]):
        ndi.affine_transform(reference[..., ch], forward, output=target[..., ch])
    matrix = affine(reference, target, multichannel=True)
    # distinguish errors on the 2x2 sub matrix and translation
    assert_array_almost_equal(matrix[:2, :2], inverse[:2, :2], decimal=1)
    assert_array_almost_equal(matrix[:2, -1], inverse[:2, -1], decimal=0)


def test_matrix_parameter_vector_conversion():
    for ndim in range(2, 5):
        p_v = np.random.rand((ndim + 1) * ndim)
        matrix = _parameter_vector_to_matrix(p_v)
        en = np.zeros(ndim + 1)
        en[-1] = 1
        p_v_2 = np.concatenate((p_v.reshape((ndim, ndim + 1)), en[np.newaxis]), axis=0)
        assert_array_equal(matrix, p_v_2)
