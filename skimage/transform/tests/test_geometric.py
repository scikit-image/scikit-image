import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from skimage.transform._geometric import _stackcopy
from skimage.transform import (estimate_transform,
                               SimilarityTransform, AffineTransform,
                               ProjectiveTransform, PolynomialTransform,
                               PiecewiseAffineTransform)


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
    # exact solution
    tform = estimate_transform('similarity', SRC[:2, :], DST[:2, :])
    assert_array_almost_equal(tform(SRC[:2, :]), DST[:2, :])
    assert_equal(tform._matrix[0, 0], tform._matrix[1, 1])
    assert_equal(tform._matrix[0, 1], - tform._matrix[1, 0])

    # over-determined
    tform2 = estimate_transform('similarity', SRC, DST)
    assert_array_almost_equal(tform2.inverse(tform2(SRC)), SRC)
    assert_equal(tform2._matrix[0, 0], tform2._matrix[1, 1])
    assert_equal(tform2._matrix[0, 1], - tform2._matrix[1, 0])

    # via estimate method
    tform3 = SimilarityTransform()
    tform3.estimate(SRC, DST)
    assert_array_almost_equal(tform3._matrix, tform2._matrix)


def test_similarity_init():
    # init with implicit parameters
    scale = 0.1
    rotation = 1
    translation = (1, 1)
    tform = SimilarityTransform(scale=scale, rotation=rotation,
                                translation=translation)
    assert_array_almost_equal(tform.scale, scale)
    assert_array_almost_equal(tform.rotation, rotation)
    assert_array_almost_equal(tform.translation, translation)

    # init with transformation matrix
    tform2 = SimilarityTransform(tform._matrix)
    assert_array_almost_equal(tform2.scale, scale)
    assert_array_almost_equal(tform2.rotation, rotation)
    assert_array_almost_equal(tform2.translation, translation)


def test_affine_estimation():
    # exact solution
    tform = estimate_transform('affine', SRC[:3, :], DST[:3, :])
    assert_array_almost_equal(tform(SRC[:3, :]), DST[:3, :])

    # over-determined
    tform2 = estimate_transform('affine', SRC, DST)
    assert_array_almost_equal(tform2.inverse(tform2(SRC)), SRC)

    # via estimate method
    tform3 = AffineTransform()
    tform3.estimate(SRC, DST)
    assert_array_almost_equal(tform3._matrix, tform2._matrix)


def test_affine_init():
    # init with implicit parameters
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

    # init with transformation matrix
    tform2 = AffineTransform(tform._matrix)
    assert_array_almost_equal(tform2.scale, scale)
    assert_array_almost_equal(tform2.rotation, rotation)
    assert_array_almost_equal(tform2.shear, shear)
    assert_array_almost_equal(tform2.translation, translation)


def test_piecewise_affine():
    tform = PiecewiseAffineTransform()
    tform.estimate(SRC, DST)
    # make sure each single affine transform is exactly estimated
    assert_array_almost_equal(tform(SRC), DST)
    assert_array_almost_equal(tform.inverse(DST), SRC)


def test_projective_estimation():
    # exact solution
    tform = estimate_transform('projective', SRC[:4, :], DST[:4, :])
    assert_array_almost_equal(tform(SRC[:4, :]), DST[:4, :])

    # over-determined
    tform2 = estimate_transform('projective', SRC, DST)
    assert_array_almost_equal(tform2.inverse(tform2(SRC)), SRC)

    # via estimate method
    tform3 = ProjectiveTransform()
    tform3.estimate(SRC, DST)
    assert_array_almost_equal(tform3._matrix, tform2._matrix)


def test_projective_init():
    tform = estimate_transform('projective', SRC, DST)
    # init with transformation matrix
    tform2 = ProjectiveTransform(tform._matrix)
    assert_array_almost_equal(tform2._matrix, tform._matrix)


def test_polynomial_estimation():
    # over-determined
    tform = estimate_transform('polynomial', SRC, DST, order=10)
    assert_array_almost_equal(tform(SRC), DST, 6)

    # via estimate method
    tform2 = PolynomialTransform()
    tform2.estimate(SRC, DST, order=10)
    assert_array_almost_equal(tform2._params, tform._params)


def test_polynomial_init():
    tform = estimate_transform('polynomial', SRC, DST, order=10)
    # init with transformation parameters
    tform2 = PolynomialTransform(tform._params)
    assert_array_almost_equal(tform2._params, tform._params)


def test_union():
    tform1 = SimilarityTransform(scale=0.1, rotation=0.3)
    tform2 = SimilarityTransform(scale=0.1, rotation=0.9)
    tform3 = SimilarityTransform(scale=0.1 ** 2, rotation=0.3 + 0.9)

    tform = tform1 + tform2

    assert_array_almost_equal(tform._matrix, tform3._matrix)


if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
