import numpy as np
try:
  import matplotlib.pyplot as plt
except ImportError:
  plt = None
from numpy.testing import assert_equal, assert_raises

from skimage.feature.util import (FeatureDetector, DescriptorExtractor,
                                  _prepare_grayscale_input_2D,
                                  _mask_border_keypoints, plot_matches)


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


@np.testing.decorators.skipif(plt is None)
def test_plot_matches():
    fig, ax = plt.subplots(nrows=1, ncols=1)

    shapes = (((10, 10), (10, 10)),
              ((10, 10), (12, 10)),
              ((10, 10), (10, 12)),
              ((10, 10), (12, 12)),
              ((12, 10), (10, 10)),
              ((10, 12), (10, 10)),
              ((12, 12), (10, 10)))

    keypoints1 = 10 * np.random.rand(10, 2)
    keypoints2 = 10 * np.random.rand(10, 2)
    idxs1 = np.random.randint(10, size=10)
    idxs2 = np.random.randint(10, size=10)
    matches = np.column_stack((idxs1, idxs2))

    for shape1, shape2 in shapes:
        img1 = np.zeros(shape1)
        img2 = np.zeros(shape2)
        plot_matches(ax, img1, img2, keypoints1, keypoints2, matches)
        plot_matches(ax, img1, img2, keypoints1, keypoints2, matches,
                     only_matches=True)
        plot_matches(ax, img1, img2, keypoints1, keypoints2, matches,
                     keypoints_color='r')
        plot_matches(ax, img1, img2, keypoints1, keypoints2, matches,
                     matches_color='r')


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
