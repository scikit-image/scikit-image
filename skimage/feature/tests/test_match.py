import numpy as np
from numpy.testing import assert_equal, assert_raises
from skimage import data
from skimage import transform as tf
from skimage.color import rgb2gray
from skimage.feature import (BRIEF, match_descriptors,
                             corner_peaks, corner_harris)


def test_binary_descriptors_unequal_descriptor_sizes_error():
    """Sizes of descriptors of keypoints to be matched should be equal."""
    descs1 = np.array([[True, True, False, True],
                     [False, True, False, True]])
    descs2 = np.array([[True, False, False, True, False],
                     [False, True, True, True, False]])
    assert_raises(ValueError, match_descriptors, descs1, descs2)


def test_binary_descriptors():
    descs1 = np.array([[True, True, False, True, True],
                     [False, True, False, True, True]])
    descs2 = np.array([[True, False, False, True, False],
                     [False, False, True, True, True]])
    matches = match_descriptors(descs1, descs2)
    assert_equal(matches, [[0, 0], [1, 1]])


def test_binary_descriptors_lena_rotation_crosscheck_false():
    """Verify matched keypoints and their corresponding masks results between
    lena image and its rotated version with the expected keypoint pairs with
    cross_check disabled."""
    img = data.lena()
    img = rgb2gray(img)
    tform = tf.SimilarityTransform(scale=1, rotation=0.15, translation=(0, 0))
    rotated_img = tf.warp(img, tform, clip=False)

    extractor = BRIEF(descriptor_size=512)

    keypoints1 = corner_peaks(corner_harris(img), min_distance=5)
    extractor.extract(img, keypoints1)
    descriptors1 = extractor.descriptors

    keypoints2 = corner_peaks(corner_harris(rotated_img), min_distance=5)
    extractor.extract(rotated_img, keypoints2)
    descriptors2 = extractor.descriptors

    matches = match_descriptors(descriptors1, descriptors2, cross_check=False)

    exp_matches1 = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
                             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                             24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                             36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46])
    exp_matches2 = np.array([33,  0, 35,  7,  1, 35,  3,  2,  3,  6,  4,  9,
                             11, 10, 28,  7,  8,  5, 31, 14, 13, 15, 21, 16,
                             16, 13, 17, 18, 19, 21, 22, 23,  0, 24,  1, 24,
                             23,  0, 26, 27, 25, 34, 28, 14, 29, 30, 21])
    assert_equal(matches[:, 0], exp_matches1)
    assert_equal(matches[:, 1], exp_matches2)


def test_binary_descriptors_lena_rotation_crosscheck_true():
    """Verify matched keypoints and their corresponding masks results between
    lena image and its rotated version with the expected keypoint pairs with
    cross_check enabled."""
    img = data.lena()
    img = rgb2gray(img)
    tform = tf.SimilarityTransform(scale=1, rotation=0.15, translation=(0, 0))
    rotated_img = tf.warp(img, tform, clip=False)

    extractor = BRIEF(descriptor_size=512)

    keypoints1 = corner_peaks(corner_harris(img), min_distance=5)
    extractor.extract(img, keypoints1)
    descriptors1 = extractor.descriptors

    keypoints2 = corner_peaks(corner_harris(rotated_img), min_distance=5)
    extractor.extract(rotated_img, keypoints2)
    descriptors2 = extractor.descriptors

    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)

    exp_matches1 = np.array([ 0,  1,  2,  4,  6,  7,  9, 10, 11, 12, 13, 15,
                               16, 17, 19, 20, 21, 24, 26, 27, 28, 29, 30, 35,
                               36, 38, 39, 40, 42, 44, 45])
    exp_matches2 = np.array([33,  0, 35,  1,  3,  2,  6,  4,  9, 11, 10,  7,
                                8,  5, 14, 13, 15, 16, 17, 18, 19, 21, 22, 24,
                                23, 26, 27, 25, 28, 29, 30])
    assert_equal(matches[:, 0], exp_matches1)
    assert_equal(matches[:, 1], exp_matches2)


def test_max_distance():
    descs1 = np.zeros((10, 128))
    descs2 = np.zeros((15, 128))

    descs1[0, :] = 1

    matches =  match_descriptors(descs1, descs2, metric='euclidean',
                                 max_distance=0.1, cross_check=False)
    assert len(matches) == 9

    matches =  match_descriptors(descs1, descs2, metric='euclidean',
                                 max_distance=np.sqrt(128.1),
                                 cross_check=False)
    assert len(matches) == 10

    matches =  match_descriptors(descs1, descs2, metric='euclidean',
                                 max_distance=0.1,
                                 cross_check=True)
    assert_equal(matches, [[1, 0]])

    matches =  match_descriptors(descs1, descs2, metric='euclidean',
                                 max_distance=np.sqrt(128.1),
                                 cross_check=True)
    assert_equal(matches, [[1, 0]])


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
