import numpy as np
import pytest

from skimage._shared._dependency_checks import has_mpl
from skimage.feature.util import (
    FeatureDetector,
    DescriptorExtractor,
    _prepare_grayscale_input_2D,
    _mask_border_keypoints,
    plot_matches,
    plot_matched_features,
)


def test_feature_detector():
    with pytest.raises(NotImplementedError):
        FeatureDetector().detect(None)


def test_descriptor_extractor():
    with pytest.raises(NotImplementedError):
        DescriptorExtractor().extract(None, None)


def test_prepare_grayscale_input_2D():
    with pytest.raises(ValueError):
        _prepare_grayscale_input_2D(np.zeros((3, 3, 3)))
    with pytest.raises(ValueError):
        _prepare_grayscale_input_2D(np.zeros((3, 1)))
    with pytest.raises(ValueError):
        _prepare_grayscale_input_2D(np.zeros((3, 1, 1)))
    _prepare_grayscale_input_2D(np.zeros((3, 3)))
    _prepare_grayscale_input_2D(np.zeros((3, 3, 1)))
    _prepare_grayscale_input_2D(np.zeros((1, 3, 3)))


def test_mask_border_keypoints():
    keypoints = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    np.testing.assert_equal(
        _mask_border_keypoints((10, 10), keypoints, 0), [1, 1, 1, 1, 1]
    )
    np.testing.assert_equal(
        _mask_border_keypoints((10, 10), keypoints, 2), [0, 0, 1, 1, 1]
    )
    np.testing.assert_equal(
        _mask_border_keypoints((4, 4), keypoints, 2), [0, 0, 1, 0, 0]
    )
    np.testing.assert_equal(
        _mask_border_keypoints((10, 10), keypoints, 5), [0, 0, 0, 0, 0]
    )
    np.testing.assert_equal(
        _mask_border_keypoints((10, 10), keypoints, 4), [0, 0, 0, 0, 1]
    )


@pytest.mark.skipif(not has_mpl, reason="Matplotlib not installed")
@pytest.mark.parametrize(
    "shapes",
    [
        ((10, 10), (10, 10)),
        ((10, 10), (12, 10)),
        ((10, 10), (10, 12)),
        ((10, 10), (12, 12)),
        ((12, 10), (10, 10)),
        ((10, 12), (10, 10)),
        ((12, 12), (10, 10)),
    ],
)
def test_plot_matches(shapes):
    from matplotlib import pyplot as plt
    from matplotlib import use

    use('Agg')

    fig, ax = plt.subplots(nrows=1, ncols=1)

    keypoints1 = 10 * np.random.rand(10, 2)
    keypoints2 = 10 * np.random.rand(10, 2)
    idxs1 = np.random.randint(10, size=10)
    idxs2 = np.random.randint(10, size=10)
    matches = np.column_stack((idxs1, idxs2))

    shape1, shape2 = shapes
    img1 = np.zeros(shape1)
    img2 = np.zeros(shape2)
    with pytest.warns(FutureWarning):
        plot_matches(ax, img1, img2, keypoints1, keypoints2, matches)
    with pytest.warns(FutureWarning):
        plot_matches(ax, img1, img2, keypoints1, keypoints2, matches, only_matches=True)
    with pytest.warns(FutureWarning):
        plot_matches(
            ax, img1, img2, keypoints1, keypoints2, matches, keypoints_color='r'
        )
    with pytest.warns(FutureWarning):
        plot_matches(ax, img1, img2, keypoints1, keypoints2, matches, matches_color='r')
    with pytest.warns(FutureWarning):
        plot_matches(
            ax, img1, img2, keypoints1, keypoints2, matches, alignment='vertical'
        )
    plt.close()


@pytest.mark.skipif(not has_mpl, reason="Matplotlib not installed")
@pytest.mark.parametrize(
    "shapes",
    [
        ((10, 10), (10, 10)),
        ((10, 10), (12, 10)),
        ((10, 10), (10, 12)),
        ((10, 10), (12, 12)),
        ((12, 10), (10, 10)),
        ((10, 12), (10, 10)),
        ((12, 12), (10, 10)),
    ],
)
def test_plot_matched_features(shapes):
    from matplotlib import pyplot as plt
    from matplotlib import use

    use('Agg')

    fig, ax = plt.subplots()

    keypoints0 = 10 * np.random.rand(10, 2)
    keypoints1 = 10 * np.random.rand(10, 2)
    idxs0 = np.random.randint(10, size=10)
    idxs1 = np.random.randint(10, size=10)
    matches = np.column_stack((idxs0, idxs1))

    shape0, shape1 = shapes
    img0 = np.zeros(shape0)
    img1 = np.zeros(shape1)
    plot_matched_features(
        img0,
        img1,
        keypoints0=keypoints0,
        keypoints1=keypoints1,
        matches=matches,
        ax=ax,
    )
    plot_matched_features(
        img0,
        img1,
        ax=ax,
        keypoints0=keypoints0,
        keypoints1=keypoints1,
        matches=matches,
        only_matches=True,
    )
    plot_matched_features(
        img0,
        img1,
        ax=ax,
        keypoints0=keypoints0,
        keypoints1=keypoints1,
        matches=matches,
        keypoints_color='r',
    )
    plot_matched_features(
        img0,
        img1,
        ax=ax,
        keypoints0=keypoints0,
        keypoints1=keypoints1,
        matches=matches,
        matches_color='r',
    )
    plot_matched_features(
        img0,
        img1,
        ax=ax,
        keypoints0=keypoints0,
        keypoints1=keypoints1,
        matches=matches,
        alignment='vertical',
    )
    plt.close()
