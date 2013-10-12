import numpy as np
from numpy.testing import assert_array_equal, assert_raises
from skimage import data
from skimage import transform as tf
from skimage.color import rgb2gray
from skimage.feature import (descriptor_brief, match_binary_descriptors, corner_peaks,
                             corner_harris)


def test_descriptor_brief_color_image_unsupported_error():
    """Brief descriptors can be evaluated on gray-scale images only."""
    img = np.zeros((20, 20, 3))
    keypoints = [[7, 5], [11, 13]]
    assert_raises(ValueError, descriptor_brief, img, keypoints)


def test_descriptor_brief_normal_mode():
    """Verify the computed BRIEF descriptors with expected for normal mode."""
    img = data.lena()
    img = rgb2gray(img)
    keypoints = corner_peaks(corner_harris(img), min_distance=5)
    descriptors, keypoints = descriptor_brief(img, keypoints[:8],
                                              descriptor_size=8)

    expected = np.array([[ True, False,  True, False,  True,  True, False, False],
                         [False, False, False, False,  True, False, False, False],
                         [ True,  True,  True,  True,  True,  True,  True,  True],
                         [ True, False,  True,  True, False,  True, False,  True],
                         [False,  True,  True,  True,  True,  True,  True,  True],
                         [ True, False, False, False, False,  True, False,  True],
                         [False,  True,  True,  True, False, False,  True, False],
                         [False, False, False, False,  True, False, False, False]], dtype=bool)

    assert_array_equal(descriptors, expected)


def test_descriptor_brief_uniform_mode():
    """Verify the computed BRIEF descriptors with expected for uniform mode."""
    img = data.lena()
    img = rgb2gray(img)
    keypoints = corner_peaks(corner_harris(img), min_distance=5)
    descriptors, keypoints = descriptor_brief(img, keypoints[:8],
                                              descriptor_size=8,
                                              mode='uniform')

    expected = np.array([[ True, False,  True, False, False,  True, False, False],
                         [False,  True, False, False,  True,  True,  True,  True],
                         [ True, False, False, False, False, False, False, False],
                         [False,  True,  True, False, False, False,  True, False],
                         [False, False, False, False, False, False,  True, False],
                         [False,  True, False, False,  True, False, False, False],
                         [False, False,  True,  True, False, False,  True,  True],
                         [ True,  True, False, False, False, False, False, False]], dtype=bool)

    assert_array_equal(descriptors, expected)


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
