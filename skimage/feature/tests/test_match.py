import numpy as np
from numpy.testing import assert_array_equal, assert_raises
from skimage import data
from skimage import transform as tf
from skimage.color import rgb2gray
from skimage.feature import (descriptor_brief, match_binary_descriptors,
                             corner_peaks, corner_harris)


def test_match_binary_descriptors_unequal_descriptor_keypoints_error():
    """Number of descriptors should be equal to the number of keypoints."""
    kp1 = np.array([[40, 50],
                    [60, 40],
                    [30, 70]])
    des1 = np.array([[True, True, False, True],
                     [False, True, False, True]])
    kp2 = np.array([[60, 50],
                    [50, 80]])
    des2 = np.array([[True, False, False, True],
                     [False, True, True, True]])
    assert_raises(ValueError, match_binary_descriptors, kp1, des1, kp2, des2)


def test_match_binary_descriptors_unequal_descriptor_sizes_error():
    """Sizes of descriptors of keypoints to be matched should be equal."""
    kp1 = np.array([[40, 50],
                    [60, 40]])
    des1 = np.array([[True, True, False, True],
                     [False, True, False, True]])
    kp2 = np.array([[60, 50],
                    [50, 80]])
    des2 = np.array([[True, False, False, True, False],
                     [False, True, True, True, False]])
    assert_raises(ValueError, match_binary_descriptors, kp1, des1, kp2, des2)


def test_match_binary_descriptors_lena_rotation_crosscheck_false():
    """Verify matched keypoints and their corresponding masks results between
    lena image and its rotated version with the expected keypoint pairs with
    cross_check disabled."""
    img = data.lena()
    img = rgb2gray(img)
    tform = tf.SimilarityTransform(scale=1, rotation=0.15, translation=(0, 0))
    rotated_img = tf.warp(img, tform)

    keypoints1 = corner_peaks(corner_harris(img), min_distance=5)
    descriptors1, keypoints1 = descriptor_brief(img, keypoints1, descriptor_size=512)

    keypoints2 = corner_peaks(corner_harris(rotated_img), min_distance=5)
    descriptors2, keypoints2 = descriptor_brief(rotated_img, keypoints2,
                                                descriptor_size=512)

    matched_keypoints, m1, m2 = match_binary_descriptors(keypoints1,
                                                         descriptors1,
                                                         keypoints2,
                                                         descriptors2,
                                                         threshold=0.13,
                                                         cross_check=False)

    expected_mask1 = np.array([11, 12, 16, 20, 24, 26, 27, 29, 35, 39, 40, 42, 45])
    expected_mask2 = np.array([ 1,  3,  0,  4,  6,  7,  8,  9, 10, 10, 11, 12, 13])
    expected = np.array([[[245, 141],
                          [221, 176]],

                         [[247, 130],
                          [225, 165]],

                         [[263, 272],
                          [219, 309]],

                         [[271, 120],
                          [250, 159]],

                         [[311, 174],
                          [282, 218]],

                         [[323, 164],
                          [294, 210]],

                         [[327, 147],
                          [301, 195]],

                         [[377, 157],
                          [349, 211]],

                         [[414,  70],
                          [399, 131]],

                         [[425,  67],
                          [399, 131]],

                         [[435, 181],
                          [403, 244]],

                         [[454, 176],
                          [423, 242]],

                         [[467, 166],
                          [437, 234]]])

    assert_array_equal(matched_keypoints, expected)
    assert_array_equal(m1, expected_mask1)
    assert_array_equal(m2, expected_mask2)


def test_match_binary_descriptors_lena_rotation_crosscheck_true():
    """Verify matched keypoints and their corresponding masks results between
    lena image and its rotated version with the expected keypoint pairs with
    cross_check enabled."""
    img = data.lena()
    img = rgb2gray(img)
    tform = tf.SimilarityTransform(scale=1, rotation=0.15, translation=(0, 0))
    rotated_img = tf.warp(img, tform)

    keypoints1 = corner_peaks(corner_harris(img), min_distance=5)
    descriptors1, keypoints1 = descriptor_brief(img, keypoints1, descriptor_size=512)

    keypoints2 = corner_peaks(corner_harris(rotated_img), min_distance=5)
    descriptors2, keypoints2 = descriptor_brief(rotated_img, keypoints2,
                                                descriptor_size=512)

    matched_keypoints, m1, m2 = match_binary_descriptors(keypoints1,
                                                         descriptors1,
                                                         keypoints2,
                                                         descriptors2,
                                                         threshold=0.13)

    expected = np.array([[[245, 141],
                          [221, 176]],

                         [[247, 130],
                          [225, 165]],

                         [[263, 272],
                          [219, 309]],

                         [[271, 120],
                          [250, 159]],

                         [[311, 174],
                          [282, 218]],

                         [[323, 164],
                          [294, 210]],

                         [[327, 147],
                          [301, 195]],

                         [[377, 157],
                          [349, 211]],

                         [[414,  70],
                          [399, 131]],

                         [[435, 181],
                          [403, 244]],

                         [[454, 176],
                          [423, 242]],

                         [[467, 166],
                          [437, 234]]])

    expected_mask1 = np.array([11, 12, 16, 20, 24, 26, 27, 29, 35, 40, 42, 45])
    expected_mask2 = np.array([ 1,  3,  0,  4,  6,  7,  8,  9, 10, 11, 12, 13])
    assert_array_equal(matched_keypoints, expected)
    assert_array_equal(m1, expected_mask1)
    assert_array_equal(m2, expected_mask2)


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
