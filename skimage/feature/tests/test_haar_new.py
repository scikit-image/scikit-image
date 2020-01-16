# File name: test_haar_new.py
# Contributor: Amazon Lab126 Multimedia team
# Date created: 01/10/2020

from random import shuffle
from itertools import chain

import pytest

import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal

from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature

from haar_fast import HaarCalculator
from haar_fast import DirectHaarCalculator
from skimage.feature import haar_like_feature_fast

def test_haar_like_feature_error():
    img = np.ones((5, 5), dtype=np.float32)
    img_ii = integral_image(img)

    feature_type = 'unknown_type'
    with pytest.raises(ValueError):
        haar_like_feature(img_ii, 0, 0, 5, 5, feature_type=feature_type)
        haar_like_feature_coord(5, 5, feature_type=feature_type)
        draw_haar_like_feature(img, 0, 0, 5, 5, feature_type=feature_type)

    feat_coord, feat_type = haar_like_feature_coord(5, 5, 'type-2-x')
    with pytest.raises(ValueError):
        haar_like_feature(img_ii, 0, 0, 5, 5, feature_type=feat_type[:3],
                          feature_coord=feat_coord)

@pytest.mark.parametrize("feature_type", ['type-2-x', 'type-2-y',
                                          'type-3-x', 'type-3-y',
                                          'type-4',
                                          ['type-2-y', 'type-3-x',
                                           'type-4']])
def test_haar_like_feature_precomputed(feature_type):
    img = np.ones((5, 5), dtype=np.int8)
    img_ii = integral_image(img)
    if isinstance(feature_type, list):
        # shuffle the index of the feature to be sure that we are output
        # the features in the same order
        shuffle(feature_type)
        feat_coord, feat_type = zip(*[haar_like_feature_coord(5, 5, feat_t)
                                      for feat_t in feature_type])
        feat_coord = np.concatenate(feat_coord)
        feat_type = np.concatenate(feat_type)
    else:
        feat_coord, feat_type = haar_like_feature_coord(5, 5, feature_type)
    haar_feature_precomputed = haar_like_feature(img_ii, 0, 0, 5, 5,
                                                 feature_type=feat_type,
                                                 feature_coord=feat_coord)
    haar_feature = haar_like_feature(img_ii, 0, 0, 5, 5, feature_type)
    assert_array_equal(haar_feature_precomputed, haar_feature)


@pytest.mark.parametrize("max_n_features,nnz_values", [(None, 46),
                                                       (1, 8)])
def test_draw_haar_like_feature(max_n_features, nnz_values):
    img = np.zeros((5, 5), dtype=np.float32)
    coord, _ = haar_like_feature_coord(5, 5, 'type-4')
    image = draw_haar_like_feature(img, 0, 0, 5, 5, coord,
                                   max_n_features=max_n_features,
                                   random_state=0)
    assert image.shape == (5, 5, 3)
    assert np.count_nonzero(image) == nnz_values

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
    haar_feature = haar_like_feature(img_ii, 0, 0, 5, 5,
                                     feature_type=feature_type)
    assert haar_feature.dtype == expected_dtype

@pytest.mark.parametrize("dtype", [np.int, np.float32, np.float64])
@pytest.mark.parametrize("dimension", [(5,5),(10,5),(5,10),(24,24)]) 
@pytest.mark.parametrize("feature_type",
                         [('type-2-x'),
                          ('type-3-x'),
                          ('type-2-y'),
                          ('type-3-y'),
                          ('type-4')])
def test_haar_like_feature(feature_type, dtype, dimension):
    img = np.random.randint(dimension[0]*dimension[1], size=dimension).astype(dtype)
    img_ii = integral_image(img)
    haar_feature = haar_like_feature(img_ii, 0, 0, dimension[1], dimension[0],
                                     feature_type=feature_type)
    img = img.reshape(1, img.shape[0], img.shape[1])

    calculator = HaarCalculator(img)
    calculator.caculate_fmap_from_type(feature_type)
    HaarResult = calculator.segmented_array
    fast_feature_value = HaarResult.get_feature()
    assert_allclose(np.sort(np.unique(haar_feature)), np.sort(np.unique(fast_feature_value)))


def test_haar_like_feature_list():
    img = np.ones((5, 5), dtype=np.int8)
    img_ii = integral_image(img)
    feature_type = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4']
    haar_list = haar_like_feature(img_ii, 0, 0, 5, 5,
                                  feature_type=feature_type)
    haar_all = haar_like_feature(img_ii, 0, 0, 5, 5)
    assert_array_equal(haar_list, haar_all)

# verify the ability of translate segment tree to coordinate
@pytest.mark.parametrize("feature_type",
                         [('type-2-x'),
                          ('type-3-x'),
                          ('type-2-y'),
                          ('type-3-y'),
                          ('type-4')])
@pytest.mark.parametrize("dimension", [(5,5), (10,10), (10,5), (5,10)]) 
def test_haar_like_feature_coord(feature_type, dimension):
    img = np.random.randint(dimension[0]*dimension[1], size=dimension).astype(int)
    img_ii = integral_image(img)

    width  = dimension[1]
    height = dimension[0]
    img = img.reshape(1, dimension[0], dimension[1])

    calculator = HaarCalculator(img)
    calculator.caculate_fmap_from_type(feature_type)
    HaarResult = calculator.segmented_array

    all_features = HaarResult.get_feature()

    feat_coord, feat_type = haar_like_feature_coord(width, height,
                                                    feature_type)

    for i in range(0, all_features.shape[1]):
        v1    = HaarResult.query_by_index(i)
        ftype = HaarResult.to_feature_type(i)
        coord = HaarResult.to_feature_coord(i)
        v2  = haar_like_feature(img_ii, 0, 0, img_ii.shape[1], img_ii.shape[0], np.array([feature_type]), coord)
        assert(v1 == v2)

@pytest.mark.parametrize("dtype", [np.int, np.float32, np.float64])
@pytest.mark.parametrize("dimension", [(5,5),(10,5),(5,10),(24,24)]) 
@pytest.mark.parametrize("feature_type",
                         [('type-2-x'),
                          ('type-3-x'),
                          ('type-2-y'),
                          ('type-3-y'),
                          ('type-4')])
def test_haar_like_feature_direct(feature_type, dtype, dimension):
    img = np.random.randint(dimension[0]*dimension[1], size=dimension).astype(dtype)
    img_ii = integral_image(img)
    haar_feature = haar_like_feature(img_ii, 0, 0, dimension[1], dimension[0],
                                     feature_type=feature_type)
    img = img.reshape(1, img.shape[0], img.shape[1])

    calculator = DirectHaarCalculator(img)
    direct_feature_value = calculator.calculate_pattern_by_type(feature_type)
    assert_allclose(np.sort(np.unique(haar_feature)), np.sort(np.unique(direct_feature_value)))


@pytest.mark.parametrize("dtype", [np.int, np.float32, np.float64])
@pytest.mark.parametrize("dimension", [(5,5),(10,5),(5,10),(24,24)]) 
@pytest.mark.parametrize("feature_type",
                         [('type-2-x'),
                          ('type-3-x'),
                          ('type-2-y'),
                          ('type-3-y'),
                          ('type-4')])
def test_haar_like_feature_direct(feature_type, dtype, dimension):
    img = np.random.randint(dimension[0]*dimension[1], size=dimension).astype(dtype)
    img_ii = integral_image(img)
    haar_feature = haar_like_feature(img_ii, 0, 0, dimension[1], dimension[0],
                                     feature_type=feature_type)
    img = img.reshape(1, img.shape[0], img.shape[1])

    calculator = DirectHaarCalculator(img)
    direct_feature_value = calculator.calculate_pattern_by_type(feature_type, flatten=True)
    assert_allclose(np.sort(np.unique(haar_feature)), np.sort(np.unique(direct_feature_value)))

@pytest.mark.parametrize("dtype", [np.int, np.float32, np.float64])
@pytest.mark.parametrize("dimension", [(5,5),(10,5),(5,10),(24,24)]) 
@pytest.mark.parametrize("feature_type",
                         [('type-2-x'),
                          ('type-3-x'),
                          ('type-2-y'),
                          ('type-3-y'),
                          ('type-4')])
def test_haar_like_feature_direct_(feature_type, dtype, dimension):
    img = np.random.randint(dimension[0]*dimension[1], size=dimension).astype(dtype)
    img_ii = integral_image(img)
    haar_feature = haar_like_feature(img_ii, 0, 0, dimension[1], dimension[0],
                                     feature_type=feature_type)
    img = img.reshape(1, img.shape[0], img.shape[1])

    calculator = DirectHaarCalculator(img)
    direct_feature_value = calculator.calculate_pattern_by_type(feature_type, flatten=True)
    assert_allclose(np.sort(np.unique(haar_feature)), np.sort(np.unique(direct_feature_value)))
