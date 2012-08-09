import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal

from skimage.transform._geometric import _stackcopy
from skimage.transform import (estimate_transform, SimilarityTransform,
                               AffineTransform, ProjectiveTransform,
                               PolynomialTransform)


SRC = np.array([
    [-12.3705, -10.5075],
    [-10.7865, 15.4305],
    [8.6985, 10.8675],
    [11.4975, -9.5715],
    [7.8435, 7.4835],
    [-5.3325, 6.5025],
    [6.7905, -6.3765],
    [-6.1695, -0.8235],
])
DST = np.array([
    [0, 0],
    [0, 5800],
    [4900, 5800],
    [4900, 0],
    [4479, 4580],
    [1176, 3660],
    [3754, 790],
    [1024, 1931],
])


def test_stackcopy():
    layers = 4
    x = np.empty((3, 3, layers))
    y = np.eye(3, 3)
    _stackcopy(x, y)
    for i in range(layers):
        assert_array_almost_equal(x[..., i], y)


def test_similarity_estimation():
    #: exact solution
    tform = estimate_transform('similarity', SRC[:2, :], DST[:2, :])
    assert_array_almost_equal(tform(SRC[:2, :]), DST[:2, :])
    assert_equal(tform._matrix[0, 0], tform._matrix[1, 1])
    assert_equal(tform._matrix[0, 1], - tform._matrix[1, 0])

    #: over-determined
    tform = estimate_transform('similarity', SRC, DST)
    assert_array_almost_equal(tform.inverse(tform(SRC)), SRC)
    assert_equal(tform._matrix[0, 0], tform._matrix[1, 1])
    assert_equal(tform._matrix[0, 1], - tform._matrix[1, 0])


def test_similarity_implicit():
    scale = 0.1
    rotation = 1
    translation = (1, 1)
    tform = SimilarityTransform(scale=scale, rotation=rotation,
                                translation=translation)
    assert_array_almost_equal(tform.scale, scale)
    assert_array_almost_equal(tform.rotation, rotation)
    assert_array_almost_equal(tform.translation, translation)


def test_affine_estimation():
    #: exact solution
    tform = estimate_transform('affine', SRC[:3, :], DST[:3, :])
    assert_array_almost_equal(tform(SRC[:3, :]), DST[:3, :])

    #: over-determined
    tform = estimate_transform('affine', SRC, DST)
    assert_array_almost_equal(tform.inverse(tform(SRC)), SRC)


def test_affine_implicit():
    scale = (0.1, 0.13)
    rotation = 1
    shear = 0.1
    translation = (1, 1)
    tform = AffineTransform(scale=scale, rotation=rotation, shear=shear,
                            translation=translation)
    assert_array_almost_equal(tform.scale, scale)
    assert_array_almost_equal(tform.rotation, rotation)
    assert_array_almost_equal(tform.shear, shear)
    assert_array_almost_equal(tform.translation, translation)


def test_projective():
    #: exact solution
    tform = estimate_transform('projective', SRC[:4, :], DST[:4, :])
    assert_array_almost_equal(tform(SRC[:4, :]), DST[:4, :])

    #: over-determined
    tform = estimate_transform('projective', SRC, DST)
    assert_array_almost_equal(tform.inverse(tform(SRC)), SRC)


def test_polynomial():
    tform = estimate_transform('polynomial', SRC, DST, order=10)
    assert_array_almost_equal(tform(SRC), DST, 6)


def test_union():
    tform1 = SimilarityTransform(scale=0.1, rotation=0.3)
    tform2 = SimilarityTransform(scale=0.1, rotation=0.9)
    tform3 = SimilarityTransform(scale=0.1 ** 2, rotation=0.3 + 0.9)

    tform = tform1 + tform2

    assert_array_almost_equal(tform._matrix, tform3._matrix)


if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
