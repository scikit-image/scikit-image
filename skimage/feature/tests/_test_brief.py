import numpy as np
from numpy.testing import assert_array_equal, assert_raises
from skimage import data
from skimage import transform as tf
from skimage.color import rgb2gray
from skimage.feature import (brief, match_keypoints_brief, corner_peaks,
                             corner_harris)


def test_brief_color_image_unsupported_error():
    """Brief descriptors can be evaluated on gray-scale images only."""
    img = np.zeros((20, 20, 3))
    keypoints = [[7, 5], [11, 13]]
    assert_raises(ValueError, brief, img, keypoints)


def test_match_keypoints_brief_lena_translation():
    """Test matched keypoints between lena image and its translated version."""
    img = data.lena()
    img = rgb2gray(img)
    img.shape
    tform = tf.SimilarityTransform(scale=1, rotation=0, translation=(15, 20))
    translated_img = tf.warp(img, tform)

    keypoints1 = corner_peaks(corner_harris(img), min_distance=5)
    descriptors1, keypoints1 = brief(img, keypoints1, descriptor_size=512)

    keypoints2 = corner_peaks(corner_harris(translated_img), min_distance=5)
    descriptors2, keypoints2 = brief(translated_img, keypoints2,
                                     descriptor_size=512)

    matched_keypoints = match_keypoints_brief(keypoints1, descriptors1,
                                              keypoints2, descriptors2,
                                              threshold=0.10)

    assert_array_equal(matched_keypoints[:, 0, :], matched_keypoints[:, 1, :] +
                       [20, 15])


def test_match_keypoints_brief_lena_rotation():
    """Verify matched keypoints result between lena image and its rotated
    version with the expected keypoint pairs."""
    img = data.lena()
    img = rgb2gray(img)
    img.shape
    tform = tf.SimilarityTransform(scale=1, rotation=0.10, translation=(0, 0))
    rotated_img = tf.warp(img, tform)

    keypoints1 = corner_peaks(corner_harris(img), min_distance=5)
    descriptors1, keypoints1 = brief(img, keypoints1, descriptor_size=512)

    keypoints2 = corner_peaks(corner_harris(rotated_img), min_distance=5)
    descriptors2, keypoints2 = brief(rotated_img, keypoints2,
                                     descriptor_size=512)

    matched_keypoints = match_keypoints_brief(keypoints1, descriptors1,
                                              keypoints2, descriptors2,
                                              threshold=0.07)

    expected = np.array([[[263, 272],
                          [234, 298]],

                         [[271, 120],
                          [258, 146]],

                         [[323, 164],
                          [305, 195]],

                         [[414,  70],
                          [405, 111]],

                         [[435, 181],
                          [415, 223]],

                         [[454, 176],
                          [435, 221]]])

    assert_array_equal(matched_keypoints, expected)


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
