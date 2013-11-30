import numpy as np
from numpy.testing import assert_equal, assert_raises
from skimage.feature.util import (FeatureDetector, DescriptorExtractor,
                                  _prepare_grayscale_input_2D,
                                  _mask_border_keypoints)


def test_feature_detector():
    assert_raises(NotImplementedError, FeatureDetector().detect, None)


def test_descriptor_extractor():
    assert_raises(NotImplementedError, DescriptorExtractor().extract,
                  None, None)


def test_prepare_grayscale_input_2D():
    assert_raises(ValueError, _prepare_grayscale_input_2D, np.zeros((3, 3, 3)))
    assert_raises(ValueError, _prepare_grayscale_input_2D, np.zeros((3, 1)))
    assert_raises(ValueError, _prepare_grayscale_input_2D, np.zeros((3, 1, 1)))
    img = _prepare_grayscale_input_2D(np.zeros((3, 3)))
    img = _prepare_grayscale_input_2D(np.zeros((3, 3, 1)))
    img = _prepare_grayscale_input_2D(np.zeros((1, 3, 3)))


def test_mask_border_keypoints():
    keypoints = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    assert_equal(_mask_border_keypoints((10, 10), keypoints, 0),
                 [1, 1, 1, 1, 1])
    assert_equal(_mask_border_keypoints((10, 10), keypoints, 2),
                 [0, 0, 1, 1, 1])
    assert_equal(_mask_border_keypoints((4, 4), keypoints, 2),
                 [0, 0, 1, 0, 0])
    assert_equal(_mask_border_keypoints((10, 10), keypoints, 5),
                 [0, 0, 0, 0, 0])
    assert_equal(_mask_border_keypoints((10, 10), keypoints, 4),
                 [0, 0, 0, 0, 1])


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
