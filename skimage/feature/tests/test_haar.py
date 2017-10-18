import pytest

import numpy as np
from numpy.testing import assert_allclose

from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import haar_like_feature_visualize


def test_haar_like_feature_error():
    img = np.ones((5, 5), dtype=np.float32)
    img_ii = integral_image(img)

    with pytest.raises(ValueError):
        haar_like_feature(img_ii, 'unknown_type')


@pytest.mark.parametrize("dtype", [np.uint8, np.int8,
                                   np.float32, np.float64])
@pytest.mark.parametrize("feature_type,shape_feature,expected_feature_value",
                         [('type-2-x', (84,), [0.]),
                          ('type-2-y', (84,), [0.]),
                          ('type-3-x', (42,), [-4., -3., -2., -1.]),
                          ('type-3-y', (42,), [-4., -3., -2., -1.]),
                          ('type-4', (36,), [0.])])
def test_haar_like_feature(feature_type, shape_feature,
                           expected_feature_value, dtype):
    # test Haar-like feature on a basic one image
    img = np.ones((5, 5), dtype=dtype)
    img_ii = integral_image(img)
    haar_feature = haar_like_feature(img_ii, feature_type=feature_type)
    assert_allclose(np.sort(np.unique(haar_feature)), expected_feature_value)


@pytest.mark.parametrize("dtype", [np.uint8, np.int8,
                                   np.float32, np.float64])
@pytest.mark.parametrize("feature_type", ['type-2-x', 'type-2-y',
                                          'type-3-x', 'type-3-y',
                                          'type-4'])
def test_haar_like_feature_fused_type(dtype, feature_type):
    # check that the input type is kept
    img = np.ones((5, 5), dtype=dtype)
    img_ii = integral_image(img)
    expected_dtype = img_ii.dtype
    # to avoid overflow, unsigned type are converted to signed
    if 'uint' in expected_dtype.name:
        expected_dtype = np.dtype(expected_dtype.name.replace('u', ''))
    haar_feature = haar_like_feature(img_ii, feature_type=feature_type)
    assert haar_feature.dtype == expected_dtype


@pytest.mark.parametrize("feature_type,height,width,expected_coord",
                         [('type-2-x', 2, 2,
                           [[[(0, 0), (1, 0)], [(0, 0), (1, 0)]],
                            [[(0, 1), (1, 1)], [(0, 1), (1, 1)]]]),
                          ('type-2-y', 2, 2,
                           [[[(0, 0), (0, 1)], [(0, 0), (0, 1)]],
                            [[(1, 0), (1, 1)], [(1, 0), (1, 1)]]]),
                          ('type-3-x', 3, 3,
                           [[[(0, 0), (0, 0), (1, 0), (1, 0), (2, 0)],
                             [(0, 0), (1, 0), (1, 0), (2, 0), (2, 0)]],
                            [[(0, 1), (0, 1), (1, 1), (1, 1), (2, 1)],
                             [(0, 1), (1, 1), (1, 1), (2, 1), (2, 1)]],
                            [[(0, 2), (0, 2), (1, 2), (1, 2), (2, 2)],
                             [(0, 2), (1, 2), (1, 2), (2, 2), (2, 2)]]]),
                          ('type-3-y', 3, 3,
                           [[[(0, 0), (0, 0), (0, 1), (0, 1), (0, 2)],
                             [(0, 0), (0, 1), (0, 1), (0, 2), (0, 2)]],
                            [[(1, 0), (1, 0), (1, 1), (1, 1), (1, 2)],
                             [(1, 0), (1, 1), (1, 1), (1, 2), (1, 2)]],
                            [[(2, 0), (2, 0), (2, 1), (2, 1), (2, 2)],
                             [(2, 0), (2, 1), (2, 1), (2, 2), (2, 2)]]]),
                          ('type-4', 2, 2,
                           [[[(0, 0)], [(0, 0)]], [[(0, 1)], [(0, 1)]],
                            [[(1, 1)], [(1, 1)]], [[(1, 0)], [(1, 0)]]])])
def test_haar_like_feature_coord(feature_type, height, width, expected_coord):
    coord = haar_like_feature_coord(feature_type, height, width)
    assert coord == expected_coord


@pytest.mark.parametrize("max_n_features,n_features", [(None, 1350),
                                                       (10, 10)])
def test_haar_like_feature_visualize(max_n_features, n_features):
    img = np.ones((5, 5), dtype=np.float32)
    img_ii = integral_image(img)
    haar_feature_coord = haar_like_feature_coord('type-2-x', 10, 10)
    feature_set = haar_like_feature_visualize(haar_feature_coord, 10, 10,
                                              max_n_features=max_n_features,
                                              random_state=0)
    assert len(feature_set) == n_features
