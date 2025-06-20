from itertools import product
import re
import textwrap

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal

from skimage.transform import (
    AffineTransform,
    EssentialMatrixTransform,
    EuclideanTransform,
    FundamentalMatrixTransform,
    PiecewiseAffineTransform,
    PolynomialTransform,
    ProjectiveTransform,
    SimilarityTransform,
    estimate_transform,
    matrix_transform,
)
from skimage.transform._geometric import (
    _GeometricTransform,
    _affine_matrix_from_vector,
    _center_and_normalize_points,
    _apply_homogeneous,
    _euler_rotation_matrix,
    TRANSFORMS,
)

SRC = np.array(
    [
        [-12.3705, -10.5075],
        [-10.7865, 15.4305],
        [8.6985, 10.8675],
        [11.4975, -9.5715],
        [7.8435, 7.4835],
        [-5.3325, 6.5025],
        [6.7905, -6.3765],
        [-6.1695, -0.8235],
    ]
)
DST = np.array(
    [
        [0, 0],
        [0, 5800],
        [4900, 5800],
        [4900, 0],
        [4479, 4580],
        [1176, 3660],
        [3754, 790],
        [1024, 1931],
    ]
)

# Transforms accepting homogeneous matrix as input.
HMAT_TFORMS = (
    FundamentalMatrixTransform,
    ProjectiveTransform,
    AffineTransform,
    EuclideanTransform,
    SimilarityTransform,
)

# Transforms allowing ND matrix.
HMAT_TFORMS_ND = (
    ProjectiveTransform,
    AffineTransform,
    EuclideanTransform,
    SimilarityTransform,
)


def test_estimate_transform():
    for tform in ('euclidean', 'similarity', 'affine', 'projective', 'polynomial'):
        estimate_transform(tform, SRC[:2, :], DST[:2, :])
    with pytest.raises(ValueError):
        estimate_transform('foobar', SRC[:2, :], DST[:2, :])


def test_matrix_transform():
    tform = AffineTransform(scale=(0.1, 0.5), rotation=2)
    assert_equal(tform(SRC), matrix_transform(SRC, tform.params))


def test_euclidean_estimation():
    # exact solution
    tform = estimate_transform('euclidean', SRC[:2, :], SRC[:2, :] + 10)
    assert_almost_equal(tform(SRC[:2, :]), SRC[:2, :] + 10)
    assert_almost_equal(tform.params[0, 0], tform.params[1, 1])
    assert_almost_equal(tform.params[0, 1], -tform.params[1, 0])

    # over-determined
    tform2 = estimate_transform('euclidean', SRC, DST)
    assert_almost_equal(tform2.inverse(tform2(SRC)), SRC)
    assert_almost_equal(tform2.params[0, 0], tform2.params[1, 1])
    assert_almost_equal(tform2.params[0, 1], -tform2.params[1, 0])

    # via estimate method
    tform3 = EuclideanTransform()
    assert tform3.estimate(SRC, DST)
    assert_almost_equal(tform3.params, tform2.params)


def test_3d_euclidean_estimation():
    src_points = np.random.rand(1000, 3)

    # Random transformation for testing
    angles = np.random.random((3,)) * 2 * np.pi - np.pi
    rotation_matrix = _euler_rotation_matrix(angles)
    translation_vector = np.random.random((3,))
    dst_points = []
    for pt in src_points:
        pt_r = pt.reshape(3, 1)
        dst = np.matmul(rotation_matrix, pt_r) + translation_vector.reshape(3, 1)
        dst = dst.reshape(3)
        dst_points.append(dst)

    dst_points = np.array(dst_points)
    # estimating the transformation
    tform = EuclideanTransform(dimensionality=3)
    assert tform.estimate(src_points, dst_points)
    estimated_rotation = tform.rotation
    estimated_translation = tform.translation
    assert_almost_equal(estimated_rotation, rotation_matrix)
    assert_almost_equal(estimated_translation, translation_vector)


def test_euclidean_init():
    # init with implicit parameters
    rotation = 1
    translation = (1, 1)
    tform = EuclideanTransform(rotation=rotation, translation=translation)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.translation, translation)

    # init with transformation matrix
    tform2 = EuclideanTransform(tform.params)
    assert_almost_equal(tform2.rotation, rotation)
    assert_almost_equal(tform2.translation, translation)

    # test special case for scale if rotation=0
    rotation = 0
    translation = (1, 1)
    tform = EuclideanTransform(rotation=rotation, translation=translation)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.translation, translation)

    # test special case for scale if rotation=90deg
    rotation = np.pi / 2
    translation = (1, 1)
    tform = EuclideanTransform(rotation=rotation, translation=translation)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.translation, translation)


def test_similarity_estimation():
    # exact solution
    tform = estimate_transform('similarity', SRC[:2, :], DST[:2, :])
    assert_almost_equal(tform(SRC[:2, :]), DST[:2, :])
    assert_almost_equal(tform.params[0, 0], tform.params[1, 1])
    assert_almost_equal(tform.params[0, 1], -tform.params[1, 0])

    # over-determined
    tform2 = estimate_transform('similarity', SRC, DST)
    assert_almost_equal(tform2.inverse(tform2(SRC)), SRC)
    assert_almost_equal(tform2.params[0, 0], tform2.params[1, 1])
    assert_almost_equal(tform2.params[0, 1], -tform2.params[1, 0])

    # via estimate method
    tform3 = SimilarityTransform()
    assert tform3.estimate(SRC, DST)
    assert_almost_equal(tform3.params, tform2.params)


def test_3d_similarity_estimation():
    src_points = np.random.rand(1000, 3)

    # Random transformation for testing
    angles = np.random.random((3,)) * 2 * np.pi - np.pi
    scale = np.random.randint(0, 20)
    rotation_matrix = _euler_rotation_matrix(angles) * scale
    translation_vector = np.random.random((3,))
    dst_points = []
    for pt in src_points:
        pt_r = pt.reshape(3, 1)
        dst = np.matmul(rotation_matrix, pt_r) + translation_vector.reshape(3, 1)
        dst = dst.reshape(3)
        dst_points.append(dst)

    dst_points = np.array(dst_points)
    # estimating the transformation
    tform = SimilarityTransform(dimensionality=3)
    assert tform.estimate(src_points, dst_points)
    estimated_rotation = tform.rotation
    estimated_translation = tform.translation
    estimated_scale = tform.scale
    assert_almost_equal(estimated_translation, translation_vector)
    assert_almost_equal(estimated_scale, scale)
    assert_almost_equal(estimated_rotation, rotation_matrix)


def test_similarity_init():
    # init with implicit parameters
    scale = 0.1
    rotation = 1
    translation = (1, 1)
    tform = SimilarityTransform(scale=scale, rotation=rotation, translation=translation)
    assert_almost_equal(tform.scale, scale)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.translation, translation)

    # init with transformation matrix
    tform2 = SimilarityTransform(tform.params)
    assert_almost_equal(tform2.scale, scale)
    assert_almost_equal(tform2.rotation, rotation)
    assert_almost_equal(tform2.translation, translation)

    # test special case for scale if rotation=0
    scale = 0.1
    rotation = 0
    translation = (1, 1)
    tform = SimilarityTransform(scale=scale, rotation=rotation, translation=translation)
    assert_almost_equal(tform.scale, scale)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.translation, translation)

    # test special case for scale if rotation=90deg
    scale = 0.1
    rotation = np.pi / 2
    translation = (1, 1)
    tform = SimilarityTransform(scale=scale, rotation=rotation, translation=translation)
    assert_almost_equal(tform.scale, scale)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.translation, translation)

    # test special case for scale where the rotation isn't exactly 90deg,
    # but very close
    scale = 1.0
    rotation = np.pi / 2
    translation = (0, 0)
    params = np.array(
        [[0, -1, 1.33226763e-15], [1, 2.22044605e-16, -1.33226763e-15], [0, 0, 1]]
    )
    tform = SimilarityTransform(params)
    assert_almost_equal(tform.scale, scale)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.translation, translation)

    # With scalar scale and 3D, we get a FutureWarning.  This is to
    # generalize the rule that dimensionality should be implied by
    # input parameters, when given.
    with pytest.warns(FutureWarning):
        tf = SimilarityTransform(scale=4, dimensionality=3)
    assert_equal(tf([[1, 1, 1]]), [[4, 4, 4]])
    # Not so if we specify some other input giving dimensionality.
    tf = SimilarityTransform(scale=4, translation=(0, 0, 0))
    assert_equal(tf([[1, 1, 1]]), [[4, 4, 4]])
    # Or if we are in 2D, by analogy to shear etc - scalar implies 2D.
    tf = SimilarityTransform(scale=4)
    assert_equal(tf([[1, 1]]), [[4, 4]])


def test_affine_estimation():
    # exact solution
    tform = estimate_transform('affine', SRC[:3, :], DST[:3, :])
    assert_almost_equal(tform(SRC[:3, :]), DST[:3, :])

    # over-determined
    tform2 = estimate_transform('affine', SRC, DST)
    assert_almost_equal(tform2.inverse(tform2(SRC)), SRC)

    # via estimate method
    tform3 = AffineTransform()
    assert tform3.estimate(SRC, DST)
    assert_almost_equal(tform3.params, tform2.params)


def test_affine_init():
    # init with implicit parameters
    scale = (0.1, 0.13)
    rotation = 1
    shear = 0.1
    translation = (1, 1)
    tform = AffineTransform(
        scale=scale, rotation=rotation, shear=shear, translation=translation
    )
    assert_almost_equal(tform.scale, scale)
    assert_almost_equal(tform.rotation, rotation)
    assert_almost_equal(tform.shear, shear)
    assert_almost_equal(tform.translation, translation)

    # init with transformation matrix
    tform2 = AffineTransform(tform.params)
    assert_almost_equal(tform2.scale, scale)
    assert_almost_equal(tform2.rotation, rotation)
    assert_almost_equal(tform2.shear, shear)
    assert_almost_equal(tform2.translation, translation)

    # scalar vs. tuple scale arguments
    assert_almost_equal(
        AffineTransform(scale=0.5).scale, AffineTransform(scale=(0.5, 0.5)).scale
    )


def test_affine_shear():
    shear = 0.1
    # expected horizontal shear transform
    cx = -np.tan(shear)
    expected = np.array([[1, cx, 0], [0, 1, 0], [0, 0, 1]])

    tform = AffineTransform(shear=shear)
    assert_almost_equal(tform.params, expected)

    shear = (1.2, 0.8)
    # expected x, y shear transform
    cx = -np.tan(shear[0])
    cy = -np.tan(shear[1])
    expected = np.array([[1, cx, 0], [cy, 1, 0], [0, 0, 1]])

    tform = AffineTransform(shear=shear)
    assert_almost_equal(tform.params, expected)


@pytest.mark.parametrize(
    'pts, params',
    product(
        (SRC, DST),
        (
            dict(scale=(4, 5), shear=(1.4, 1.8), rotation=0.4, translation=(10, 12)),
            dict(
                scale=(-0.5, 3), shear=(-0.3, -0.1), rotation=1.4, translation=(-4, 3)
            ),
        ),
    ),
)
def test_affine_params(pts, params):
    # Test AffineTransform against docstring algorithm.
    out = AffineTransform(**params)(pts)
    docstr_out = _apply_aff_2d(pts, **params)
    assert np.allclose(out, docstr_out)


def _apply_aff_2d(pts, scale, rotation, shear, translation):
    # Algorithm from AffineTransform docstring.
    x, y = pts.T
    sx, sy = scale
    shear_x, shear_y = shear
    translation_x, translation_y = translation
    cos, tan, sin = np.cos, np.tan, np.sin
    X = (
        sx * x * (cos(rotation) + tan(shear_y) * sin(rotation))
        - sy * y * (tan(shear_x) * cos(rotation) + sin(rotation))
        + translation_x
    )
    Y = (
        sx * x * (sin(rotation) - tan(shear_y) * cos(rotation))
        - sy * y * (tan(shear_x) * sin(rotation) - cos(rotation))
        + translation_y
    )
    return np.stack((X, Y), axis=1)


def test_piecewise_affine():
    tform = PiecewiseAffineTransform()
    assert tform.estimate(SRC, DST)
    # make sure each single affine transform is exactly estimated
    assert_almost_equal(tform(SRC), DST)
    assert_almost_equal(tform.inverse(DST), SRC)


def test_fundamental_matrix_estimation():
    src = np.array(
        [
            1.839035,
            1.924743,
            0.543582,
            0.375221,
            0.473240,
            0.142522,
            0.964910,
            0.598376,
            0.102388,
            0.140092,
            15.994343,
            9.622164,
            0.285901,
            0.430055,
            0.091150,
            0.254594,
        ]
    ).reshape(-1, 2)
    dst = np.array(
        [
            1.002114,
            1.129644,
            1.521742,
            1.846002,
            1.084332,
            0.275134,
            0.293328,
            0.588992,
            0.839509,
            0.087290,
            1.779735,
            1.116857,
            0.878616,
            0.602447,
            0.642616,
            1.028681,
        ]
    ).reshape(-1, 2)

    tform = estimate_transform('fundamental', src, dst)

    # Reference values obtained using COLMAP SfM library.
    tform_ref = np.array(
        [
            [-0.217859, 0.419282, -0.0343075],
            [-0.0717941, 0.0451643, 0.0216073],
            [0.248062, -0.429478, 0.0221019],
        ]
    )
    assert_almost_equal(tform.params, tform_ref, 6)


def test_fundamental_matrix_residuals():
    essential_matrix_tform = EssentialMatrixTransform(
        rotation=np.eye(3), translation=np.array([1, 0, 0])
    )
    tform = FundamentalMatrixTransform()
    tform.params = essential_matrix_tform.params
    src = np.array([[0, 0], [0, 0], [0, 0]])
    dst = np.array([[2, 0], [2, 1], [2, 2]])
    assert_almost_equal(tform.residuals(src, dst) ** 2, [0, 0.5, 2])


@pytest.mark.parametrize('array_like_input', [False, True])
def test_fundamental_matrix_forward(array_like_input):
    if array_like_input:
        rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        translation = (1, 0, 0)
    else:
        rotation = np.eye(3)
        translation = np.array([1, 0, 0])
    essential_matrix_tform = EssentialMatrixTransform(
        rotation=rotation, translation=translation
    )
    if array_like_input:
        params = [list(p) for p in essential_matrix_tform.params]
    else:
        params = essential_matrix_tform.params
    tform = FundamentalMatrixTransform(matrix=params)
    src = np.array([[0, 0], [0, 1], [1, 1]])
    assert_almost_equal(tform(src), [[0, -1, 0], [0, -1, 1], [0, -1, 1]])


def test_fundamental_matrix_inverse():
    essential_matrix_tform = EssentialMatrixTransform(
        rotation=np.eye(3), translation=np.array([1, 0, 0])
    )
    tform = FundamentalMatrixTransform()
    tform.params = essential_matrix_tform.params
    src = np.array([[0, 0], [0, 1], [1, 1]])
    assert_almost_equal(tform.inverse(src), [[0, 1, 0], [0, 1, -1], [0, 1, -1]])


def test_fundamental_matrix_inverse_estimation():
    src = np.array(
        [
            1.839035,
            1.924743,
            0.543582,
            0.375221,
            0.473240,
            0.142522,
            0.964910,
            0.598376,
            0.102388,
            0.140092,
            15.994343,
            9.622164,
            0.285901,
            0.430055,
            0.091150,
            0.254594,
        ]
    ).reshape(-1, 2)

    dst = np.array(
        [
            1.002114,
            1.129644,
            1.521742,
            1.846002,
            1.084332,
            0.275134,
            0.293328,
            0.588992,
            0.839509,
            0.087290,
            1.779735,
            1.116857,
            0.878616,
            0.602447,
            0.642616,
            1.028681,
        ]
    ).reshape(-1, 2)

    # Inverse of (src -> dst) transform should be equivalent to
    # (dst -> src) transformation
    tform = estimate_transform('fundamental', src, dst)
    tform_inv = estimate_transform('fundamental', dst, src)

    np.testing.assert_array_almost_equal(tform.inverse.params, tform_inv.params)


def test_fundamental_matrix_epipolar_projection():
    src = np.array(
        [
            1.839035,
            1.924743,
            0.543582,
            0.375221,
            0.473240,
            0.142522,
            0.964910,
            0.598376,
            0.102388,
            0.140092,
            15.994343,
            9.622164,
            0.285901,
            0.430055,
            0.091150,
            0.254594,
        ]
    ).reshape(-1, 2)

    dst = np.array(
        [
            1.002114,
            1.129644,
            1.521742,
            1.846002,
            1.084332,
            0.275134,
            0.293328,
            0.588992,
            0.839509,
            0.087290,
            1.779735,
            1.116857,
            0.878616,
            0.602447,
            0.642616,
            1.028681,
        ]
    ).reshape(-1, 2)

    tform = estimate_transform('fundamental', src, dst)

    # calculate x' F x for each coordinate; should be close to zero
    p = np.abs(np.sum(np.column_stack((dst, np.ones(len(dst)))) * tform(src), axis=1))
    assert np.all(p < 0.01)


def test_essential_matrix_init():
    r = np.eye(3)
    t = np.array([0, 0, 1])
    tform = EssentialMatrixTransform(rotation=r, translation=t)
    assert_equal(tform.params, np.array([0, -1, 0, 1, 0, 0, 0, 0, 0]).reshape(3, 3))
    t2 = np.array([0, 0, 2])
    with pytest.raises(ValueError, match="Translation vector must have unit length"):
        EssentialMatrixTransform(rotation=r, translation=t2)
    with pytest.raises(ValueError, match="Rotation matrix must have unit determinant"):
        EssentialMatrixTransform(rotation=np.eye(3)[::-1], translation=t)
    r2 = r[[2, 0, 1]]
    tform = EssentialMatrixTransform(rotation=r2, translation=t)
    assert_equal(tform.params, [[-1, 0, 0], [0, 0, 1], [0, 0, 0]])
    with pytest.raises(ValueError):
        EssentialMatrixTransform(matrix=np.zeros((3, 2)), translation=t)
    with pytest.raises(ValueError):
        EssentialMatrixTransform(rotation=np.zeros((3, 2)), translation=t)
    with pytest.raises(ValueError):
        EssentialMatrixTransform(rotation=np.zeros((3, 3)), translation=t)
    # Both must be specified.
    with pytest.raises(ValueError):
        EssentialMatrixTransform(rotation=np.eye(3))
    # Dimensionality must match.
    with pytest.raises(ValueError):
        EssentialMatrixTransform(rotation=np.eye(3), translation=[1, 0])


def test_essential_matrix_estimation():
    src = np.array(
        [
            1.839035,
            1.924743,
            0.543582,
            0.375221,
            0.473240,
            0.142522,
            0.964910,
            0.598376,
            0.102388,
            0.140092,
            15.994343,
            9.622164,
            0.285901,
            0.430055,
            0.091150,
            0.254594,
        ]
    ).reshape(-1, 2)
    dst = np.array(
        [
            1.002114,
            1.129644,
            1.521742,
            1.846002,
            1.084332,
            0.275134,
            0.293328,
            0.588992,
            0.839509,
            0.087290,
            1.779735,
            1.116857,
            0.878616,
            0.602447,
            0.642616,
            1.028681,
        ]
    ).reshape(-1, 2)

    tform = estimate_transform('essential', src, dst)

    # Reference values obtained using COLMAP SfM library.
    tform_ref = np.array(
        [
            [-0.0811666, 0.255449, -0.0478999],
            [-0.192392, -0.0531675, 0.119547],
            [0.177784, -0.22008, -0.015203],
        ]
    )
    assert_almost_equal(tform.params, tform_ref, 6)


def test_essential_matrix_forward():
    tform = EssentialMatrixTransform(
        rotation=np.eye(3), translation=np.array([1, 0, 0])
    )
    src = np.array([[0, 0], [0, 1], [1, 1]])
    assert_almost_equal(tform(src), [[0, -1, 0], [0, -1, 1], [0, -1, 1]])


def test_essential_matrix_inverse():
    tform = EssentialMatrixTransform(
        rotation=np.eye(3), translation=np.array([1, 0, 0])
    )
    src = np.array([[0, 0], [0, 1], [1, 1]])
    assert_almost_equal(tform.inverse(src), [[0, 1, 0], [0, 1, -1], [0, 1, -1]])


def test_essential_matrix_residuals():
    tform = EssentialMatrixTransform(
        rotation=np.eye(3), translation=np.array([1, 0, 0])
    )
    src = np.array([[0, 0], [0, 0], [0, 0]])
    dst = np.array([[2, 0], [2, 1], [2, 2]])
    assert_almost_equal(tform.residuals(src, dst) ** 2, [0, 0.5, 2])


def test_projective_estimation():
    # exact solution
    tform = estimate_transform('projective', SRC[:4, :], DST[:4, :])
    assert_almost_equal(tform(SRC[:4, :]), DST[:4, :])

    # over-determined
    tform2 = estimate_transform('projective', SRC, DST)
    assert_almost_equal(tform2.inverse(tform2(SRC)), SRC)

    # via estimate method
    tform3 = ProjectiveTransform()
    assert tform3.estimate(SRC, DST)
    assert_almost_equal(tform3.params, tform2.params)


def test_projective_weighted_estimation():
    # Exact solution with same points, and unity weights
    tform = estimate_transform('projective', SRC[:4, :], DST[:4, :])
    tform_w = estimate_transform('projective', SRC[:4, :], DST[:4, :], np.ones(4))
    assert_almost_equal(tform.params, tform_w.params)

    # Over-determined solution with same points, and unity weights
    tform = estimate_transform('projective', SRC, DST)
    tform_w = estimate_transform('projective', SRC, DST, np.ones(SRC.shape[0]))
    assert_almost_equal(tform.params, tform_w.params)

    # Repeating a point, but setting its weight small, should give nearly
    # the same result.
    point_weights = np.ones(SRC.shape[0] + 1)
    point_weights[0] = 1.0e-15
    tform1 = estimate_transform('projective', SRC, DST)
    tform2 = estimate_transform(
        'projective',
        SRC[np.arange(-1, SRC.shape[0]), :],
        DST[np.arange(-1, SRC.shape[0]), :],
        point_weights,
    )
    assert_almost_equal(tform1.params, tform2.params, decimal=3)


@pytest.mark.parametrize('array_like_input', [False, True])
def test_projective_init(array_like_input):
    tform = estimate_transform('projective', SRC, DST)
    # init with transformation matrix
    if array_like_input:
        params = [list(p) for p in tform.params]
    else:
        params = tform.params
    tform2 = ProjectiveTransform(params)
    assert_almost_equal(tform2.params, tform.params)


def test_polynomial_estimation():
    # over-determined
    tform = estimate_transform('polynomial', SRC, DST, order=10)
    assert_almost_equal(tform(SRC), DST, 6)

    # via estimate method
    tform2 = PolynomialTransform()
    assert tform2.estimate(SRC, DST, order=10)
    assert_almost_equal(tform2.params, tform.params)


def test_polynomial_weighted_estimation():
    # Over-determined solution with same points, and unity weights
    tform = estimate_transform('polynomial', SRC, DST, order=10)
    tform_w = estimate_transform(
        'polynomial', SRC, DST, order=10, weights=np.ones(SRC.shape[0])
    )
    assert_almost_equal(tform.params, tform_w.params)

    # Repeating a point, but setting its weight small, should give nearly
    # the same result.
    point_weights = np.ones(SRC.shape[0] + 1)
    point_weights[0] = 1.0e-15
    tform1 = estimate_transform('polynomial', SRC, DST, order=10)
    tform2 = estimate_transform(
        'polynomial',
        SRC[np.arange(-1, SRC.shape[0]), :],
        DST[np.arange(-1, SRC.shape[0]), :],
        order=10,
        weights=point_weights,
    )
    assert_almost_equal(tform1.params, tform2.params, decimal=4)


@pytest.mark.parametrize('array_like_input', [False, True])
def test_polynomial_init(array_like_input):
    tform = estimate_transform('polynomial', SRC, DST, order=10)
    # Init with transformation parameters.
    if array_like_input:
        params = [list(p) for p in tform.params]
    else:
        params = tform.params
    tform2 = PolynomialTransform(params)
    assert_almost_equal(tform2.params, tform.params)
    # Can't specify scalar params.
    with pytest.raises(ValueError):
        _ = PolynomialTransform(0)
    # Parameters must be (2, N).
    for inp in (np.eye(3), np.zeros(3)):
        with pytest.raises(ValueError):
            _ = PolynomialTransform(inp)
    # Transform always 2D.
    for d in (1, 3, 4):
        with pytest.raises(NotImplementedError):
            _ = PolynomialTransform(dimensionality=d)
        with pytest.raises(NotImplementedError):
            _ = PolynomialTransform.identity(d)


def test_polynomial_default_order():
    tform = estimate_transform('polynomial', SRC, DST)
    tform2 = estimate_transform('polynomial', SRC, DST, order=2)
    assert_almost_equal(tform2.params, tform.params)


def test_polynomial_inverse():
    with pytest.raises(NotImplementedError):
        PolynomialTransform().inverse(0)


def test_union():
    tform1 = SimilarityTransform(scale=0.1, rotation=0.3)
    tform2 = SimilarityTransform(scale=0.1, rotation=0.9)
    tform3 = SimilarityTransform(scale=0.1**2, rotation=0.3 + 0.9)
    tform = tform1 + tform2
    assert_almost_equal(tform.params, tform3.params)

    tform1 = AffineTransform(scale=(0.1, 0.1), rotation=0.3)
    tform2 = SimilarityTransform(scale=0.1, rotation=0.9)
    tform3 = SimilarityTransform(scale=0.1**2, rotation=0.3 + 0.9)
    tform = tform1 + tform2
    assert_almost_equal(tform.params, tform3.params)
    assert tform.__class__ == ProjectiveTransform

    tform = AffineTransform(scale=(0.1, 0.1), rotation=0.3)
    assert_almost_equal((tform + tform.inverse).params, np.eye(3))

    tform1 = SimilarityTransform(scale=0.1, rotation=0.3)
    tform2 = SimilarityTransform(scale=0.1, rotation=0.9)
    tform3 = SimilarityTransform(scale=0.1 * 1 / 0.1, rotation=0.3 - 0.9)
    tform = tform1 + tform2.inverse
    assert_almost_equal(tform.params, tform3.params)


def test_union_differing_types():
    tform1 = SimilarityTransform()
    tform2 = PolynomialTransform()
    with pytest.raises(TypeError):
        tform1.__add__(tform2)


@pytest.mark.parametrize(
    "tform",
    [
        ProjectiveTransform(matrix=np.random.rand(3, 3)),
        AffineTransform(scale=(0.1, 0.1), rotation=0.3),
        EuclideanTransform(rotation=0.9, translation=(5, 5)),
        SimilarityTransform(scale=0.1, rotation=0.9),
        EssentialMatrixTransform(
            rotation=np.eye(3), translation=(1 / np.sqrt(2), 1 / np.sqrt(2), 0)
        ),
        FundamentalMatrixTransform(
            matrix=EssentialMatrixTransform(
                rotation=np.eye(3), translation=(1 / np.sqrt(2), 1 / np.sqrt(2), 0)
            ).params
        ),
        ((t := PiecewiseAffineTransform()).estimate(SRC, DST) and t),
    ],
)
def test_inverse_all_transforms(tform):
    assert isinstance(tform.inverse, type(tform))
    if hasattr(tform, 'params'):
        assert_almost_equal(tform.inverse.inverse.params, tform.params)
    else:
        assert isinstance(tform, PiecewiseAffineTransform)
    assert_almost_equal(tform.inverse.inverse(SRC), tform(SRC))
    # Test addition with inverse, not implemented for all
    if not isinstance(
        tform,
        (
            EssentialMatrixTransform,
            FundamentalMatrixTransform,
            PiecewiseAffineTransform,
        ),
    ):
        assert_almost_equal((tform + tform.inverse)(SRC), SRC)
        assert_almost_equal((tform.inverse + tform)(SRC), SRC)


@pytest.mark.parametrize('tform_class', TRANSFORMS.values())
def test_identity(tform_class):
    if tform_class is PiecewiseAffineTransform:
        return  # Identity transform unusable.
    rng = np.random.default_rng()
    allows_nd = tform_class in HMAT_TFORMS_ND
    for ndim in (2, 3, 4, 5) if allows_nd else (2,):
        src = rng.normal(size=(10, ndim))
        t = tform_class.identity(ndim)
        if isinstance(t, FundamentalMatrixTransform):
            out = np.hstack((src, np.ones((len(src), 1))))
        else:
            out = src
        assert np.allclose(t(src), out)


def test_geometric_tform():
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        _GeometricTransform()

    # See gh-3926 for discussion details
    for i in range(20):
        # Generate random Homography
        H = np.random.rand(3, 3) * 100
        H[2, H[2] == 0] += np.finfo(float).eps
        H /= H[2, 2]

        # Craft some src coords
        src = np.array(
            [
                [(H[2, 1] + 1) / -H[2, 0], 1],
                [1, (H[2, 0] + 1) / -H[2, 1]],
                [1, 1],
            ]
        )
        # Prior to gh-3926, under the above circumstances,
        # destination coordinates could be returned with nan/inf values.
        tform = ProjectiveTransform(H)  # Construct the transform
        dst = tform(src)  # Obtain the dst coords
        # Ensure dst coords are finite numeric values
        assert np.isfinite(dst).all()


def test_degenerate():
    src = dst = np.zeros((10, 2))

    tform = SimilarityTransform()
    assert not tform.estimate(src, dst)
    assert np.all(np.isnan(tform.params))

    tform = EuclideanTransform()
    assert not tform.estimate(src, dst)
    assert np.all(np.isnan(tform.params))

    tform = AffineTransform()
    assert not tform.estimate(src, dst)
    assert np.all(np.isnan(tform.params))

    tform = ProjectiveTransform()
    assert not tform.estimate(src, dst)
    assert np.all(np.isnan(tform.params))

    # See gh-3926 for discussion details
    tform = ProjectiveTransform()
    for i in range(20):
        # Some random coordinates
        src = np.random.rand(4, 2) * 100
        dst = np.random.rand(4, 2) * 100

        # Degenerate the case by arranging points on a single line
        src[:, 1] = np.random.rand()
        # Prior to gh-3926, under the above circumstances,
        # a transform could be returned with nan values.
        assert not tform.estimate(src, dst) or np.isfinite(tform.params).all()

    src = np.array([[0, 2, 0], [0, 2, 0], [0, 4, 0]])
    dst = np.array([[0, 1, 0], [0, 1, 0], [0, 3, 0]])
    tform = AffineTransform()
    assert not tform.estimate(src, dst)
    # Prior to gh-6207, the above would set the parameters as the identity.
    assert np.all(np.isnan(tform.params))

    # The tessellation on the following points produces one degenerate affine
    # warp within PiecewiseAffineTransform.
    src = np.asarray(
        [
            [0, 192, 256],
            [0, 256, 256],
            [5, 0, 192],
            [5, 64, 0],
            [5, 64, 64],
            [5, 64, 256],
            [5, 192, 192],
            [5, 256, 256],
            [0, 192, 256],
        ]
    )

    dst = np.asarray(
        [
            [0, 142, 206],
            [0, 206, 206],
            [5, -50, 142],
            [5, 14, 0],
            [5, 14, 64],
            [5, 14, 206],
            [5, 142, 142],
            [5, 206, 206],
            [0, 142, 206],
        ]
    )
    tform = PiecewiseAffineTransform()
    assert not tform.estimate(src, dst)
    assert np.all(np.isnan(tform.affines[4].params))  # degenerate affine
    for idx, affine in enumerate(tform.affines):
        if idx != 4:
            assert not np.all(np.isnan(affine.params))
    for affine in tform.inverse_affines:
        assert not np.all(np.isnan(affine.params))


def test_normalize_points():
    mat, normed = _center_and_normalize_points(SRC)
    assert np.allclose(np.mean(normed, axis=0), 0)
    assert np.allclose(normed, _apply_homogeneous(mat, SRC))


def test_normalize_degenerate_points():
    """Return nan matrix *of appropriate size* when point is repeated."""
    pts = np.array([[73.42834308, 94.2977623]] * 3)
    mat, pts_tf = _center_and_normalize_points(pts)
    assert np.all(np.isnan(mat))
    assert np.all(np.isnan(pts_tf))
    assert mat.shape == (3, 3)
    assert pts_tf.shape == pts.shape


def test_projective_repr():
    tform = ProjectiveTransform()
    want = (
        re.escape(
            textwrap.dedent(
                '''
        <ProjectiveTransform(matrix=
            [[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]]) at
        '''
            ).strip()
        )
        + ' 0x[a-f0-9]+'
        + re.escape('>')
    )
    # Hack the escaped regex to allow whitespace before each number for
    # compatibility with different numpy versions.
    want = want.replace('0\\.', ' *0\\.')
    want = want.replace('1\\.', ' *1\\.')
    assert re.match(want, repr(tform))


def test_projective_str():
    tform = ProjectiveTransform()
    want = re.escape(
        textwrap.dedent(
            '''
        <ProjectiveTransform(matrix=
            [[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]])>
        '''
        ).strip()
    )
    # Hack the escaped regex to allow whitespace before each number for
    # compatibility with different numpy versions.
    want = want.replace('0\\.', ' *0\\.')
    want = want.replace('1\\.', ' *1\\.')
    assert re.match(want, str(tform))


def _assert_least_squares(tf, src, dst):
    baseline = np.sum((tf(src) - dst) ** 2)
    for i in range(tf.params.size):
        for update in [0.001, -0.001]:
            params = np.copy(tf.params)
            params.flat[i] += update
            new_tf = tf.__class__(matrix=params)
            new_ssq = np.sum((new_tf(src) - dst) ** 2)
            assert new_ssq > baseline


@pytest.mark.parametrize('array_like_input', [False, True])
def test_estimate_affine_3d(array_like_input):
    ndim = 3
    src = np.random.random((25, ndim)) * 2 ** np.arange(7, 7 + ndim)
    matrix = np.array(
        [
            [4.8, 0.1, 0.2, 25],
            [0.0, 1.0, 0.1, 30],
            [0.0, 0.0, 1.0, -2],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    if array_like_input:
        # list of lists for matrix and src coords
        src = [list(c) for c in src]
        matrix = [list(c) for c in matrix]

    tf = AffineTransform(matrix=matrix)
    dst = tf(src)
    dst_noisy = dst + np.random.random((25, ndim))
    if array_like_input:
        # list of lists for destination coords
        dst = [list(c) for c in dst]
    tf2 = AffineTransform(dimensionality=ndim)
    assert tf2.estimate(src, dst_noisy)
    # we check rot/scale/etc more tightly than translation because translation
    # estimation is on the 1 pixel scale
    matrix = np.asarray(matrix)
    assert_almost_equal(tf2.params[:, :-1], matrix[:, :-1], decimal=2)
    assert_almost_equal(tf2.params[:, -1], matrix[:, -1], decimal=0)
    _assert_least_squares(tf2, src, dst_noisy)


def test_fundamental_3d_not_implemented():
    with pytest.raises(NotImplementedError):
        _ = FundamentalMatrixTransform(dimensionality=3)
    with pytest.raises(NotImplementedError):
        _ = FundamentalMatrixTransform(np.eye(4))


def test_array_protocol():
    mat = np.eye(4)
    tf = ProjectiveTransform(mat)
    assert_equal(np.array(tf), mat)
    assert_equal(np.array(tf, dtype=int), mat.astype(int))


def test_affine_transform_from_linearized_parameters():
    mat = np.concatenate((np.random.random((3, 4)), np.eye(4)[-1:]), axis=0)
    v = mat[:-1].ravel()
    mat_from_v = _affine_matrix_from_vector(v)
    tf = AffineTransform(matrix=mat_from_v)
    assert_equal(np.array(tf), mat)
    # incorrect number of parameters
    with pytest.raises(ValueError):
        _ = _affine_matrix_from_vector(v[:-1])
    with pytest.raises(ValueError):
        _ = AffineTransform(matrix=v[:-1])


EG_OPS = dict(scale=(4, 5), shear=(1.4, 1.8), rotation=0.4, translation=(10, 12))


@pytest.mark.parametrize(
    'tform_class, op_order',
    (
        (AffineTransform, ('scale', 'shear', 'rotation', 'translation')),
        (EuclideanTransform, ('rotation', 'translation')),
        (SimilarityTransform, ('scale', 'rotation', 'translation')),
    ),
)
def test_transform_order(tform_class, op_order):
    # Test transforms are applied in order stated.
    ops = [(k, EG_OPS[k]) for k in op_order]
    part_xforms = [tform_class(**{k: v}) for k, v in ops]
    full_xform = tform_class(**dict(ops))
    # Assemble affine transform via matrices.
    out = np.eye(3)
    for tf in part_xforms:
        out = tf @ out
    assert np.allclose(full_xform.params, out)


# AffineTransform only allows 2D implicit parameters.
@pytest.mark.parametrize(
    'inp',
    (
        dict(scale=5, dimensionality=3),
        dict(scale=(5, 5, 5), dimensionality=3),
        dict(scale=(5, 5, 5)),
        dict(shear=(0.1, 0.2, 0.3)),
        dict(rotation=(0.1, 0.2)),
        dict(translation=1),
        dict(translation=(1, 2, 3)),
    ),
)
def test_affine_params_nD_error(inp):
    with pytest.raises(ValueError):
        _ = AffineTransform(**inp)


def test_euler_rotation():
    for v, angles, expected in (
        ([0, 10, 0], np.radians([90, 45, 45]), [-5, -5, 7.1]),
        ([-1, 7, -2], np.radians([-10, 23, -25]), [1.1, 6.2, -3.8]),
    ):
        R = _euler_rotation_matrix(angles)
        assert_almost_equal(R @ v, expected, decimal=1)


def _from_matvec(mat, vec):
    mat, vec = (np.array(p) for p in (mat, vec))
    d = mat.shape[0]
    out = np.eye(d + 1)
    out[:-1, :-1] = mat
    out[:-1, -1] = vec
    return out


@pytest.mark.parametrize('tform_class', (EuclideanTransform, SimilarityTransform))
def test_euclidean_param_defaults(tform_class):
    # 2D rotation is 0 when only translation is given
    tf = tform_class(translation=(5, 5))
    assert np.array(tf)[0, 1] == 0
    # off diagonals are 0 when only translation is given
    tf = tform_class(translation=(4, 5, 9), dimensionality=3)
    assert_equal(np.array(tf)[[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], 0)
    # Specifying translations for D>3 is supported.
    tf = tform_class(translation=(5, 6, 7, 8))
    assert_equal(tf.params, _from_matvec(np.eye(4), (5, 6, 7, 8)))
    # But must match the dimensionality.
    with pytest.raises(ValueError):
        _ = tform_class(translation=(5, 6, 7, 8), dimensionality=3)
    # Incorrect number of angles (must be 1 or 3
    with pytest.raises(ValueError):
        _ = tform_class(rotation=(4, 8))
    with pytest.raises(ValueError):
        _ = tform_class(rotation=(4, 8, 2, 4))
    # translation is 0 when rotation is given
    tf = tform_class(rotation=np.pi * np.arange(3), dimensionality=3)
    assert_equal(np.array(tf)[:-1, 3], 0)


def test_euler_angle_consistency():
    angles = np.random.random((3,)) * 2 * np.pi - np.pi
    euclid = EuclideanTransform(rotation=angles, dimensionality=3)
    similar = SimilarityTransform(rotation=angles, dimensionality=3)
    assert_array_almost_equal(euclid, similar)


def test_2D_only_implementations():
    with pytest.raises(NotImplementedError):
        _ = PolynomialTransform(dimensionality=3)
    tf = AffineTransform(dimensionality=3)
    with pytest.raises(NotImplementedError):
        _ = tf.rotation
    with pytest.raises(NotImplementedError):
        _ = tf.shear


@pytest.mark.parametrize('tform_class', HMAT_TFORMS)
def test_kw_only_params(tform_class):
    # Check only matrix can be passed as positional arg.
    with pytest.raises(TypeError):
        tform_class(None, None)


def test_kw_only_emt():
    # Check all parameters are keyword-only for EssentialMatrixTransform.
    with pytest.raises(TypeError):
        EssentialMatrixTransform(None)


@pytest.mark.parametrize('tform_class', HMAT_TFORMS)
def test_init_contract_dims(tform_class):
    allows_nd = tform_class in HMAT_TFORMS_ND
    # 2D identity is default.
    for tf in (
        tform_class(),
        tform_class(dimensionality=2),
        tform_class.identity(),
        tform_class.identity(None),
        tform_class(None),
    ):
        assert_equal(tf.params, np.eye(3))
    ok_dims = (2, 3, 4, 5) if allows_nd else (2,)
    for d in ok_dims:
        h_d = d + 1
        # Identity for acceptable dimensions.
        for tf in (
            tform_class(dimensionality=d),
            tform_class.identity(d),
            tform_class.identity(dimensionality=d),
        ):
            assert_equal(tf.params, np.eye(h_d))
        # Wrong shape for given dimensions.
        for matrix in np.eye(h_d)[:-1], np.eye(h_d)[:, :-1]:
            with pytest.raises(ValueError):
                tform_class(matrix)
    err_dims = (1,) if allows_nd else (1, 3, 4, 5)
    for matrix in [np.eye(d + 1) for d in err_dims]:
        with pytest.raises(NotImplementedError):
            tform_class(matrix)
    # Test vector matrix input invalid.
    with pytest.raises(ValueError):
        tform_class(np.zeros((2, 3)))


def test_broadcasting():
    # Scalar scale broadcasts.
    translation = [3, 4, 5]
    tf = SimilarityTransform(scale=2, translation=translation)
    assert_equal(tf.params, _from_matvec(np.eye(3) * 2, translation))
    # Translation does broadcast.
    # 2D.
    tf = SimilarityTransform(scale=2, translation=10)
    assert_equal(tf.params, _from_matvec(np.eye(2) * 2, [10, 10]))
    # 3D.
    tf = SimilarityTransform(scale=[2, 3, 4], translation=10)
    assert_equal(tf.params, _from_matvec(np.diag([2, 3, 4]), [10] * 3))
    # Scalar rotation does not broadcast.
    for tf_class in SimilarityTransform, EuclideanTransform:
        with pytest.raises(ValueError):
            tf_class(rotation=0.2, translation=translation)
