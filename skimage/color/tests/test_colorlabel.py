import itertools
import pytest

import numpy as np
from skimage.color.colorlabel import label2rgb

from skimage._shared import testing
from skimage._shared.testing import (assert_array_almost_equal,
                                     assert_array_equal, assert_warns,
                                     assert_no_warnings)


def test_deprecation_warning():

    image = np.ones((3, 3))
    label = np.ones((3, 3))

    with pytest.warns(FutureWarning) as record:
        label2rgb(image, label)

    expected_msg = "The new recommended value"

    assert str(record[0].message).startswith(expected_msg)


def test_shape_mismatch():
    image = np.ones((3, 3))
    label = np.ones((2, 2))
    with testing.raises(ValueError):
        label2rgb(image, label, bg_label=-1)


def test_wrong_kind():
    label = np.ones((3, 3))
    # Must not raise an error.
    label2rgb(label, bg_label=-1)
    # kind='foo' is wrong.
    with testing.raises(ValueError):
        label2rgb(label, kind='foo', bg_label=-1)


def test_uint_image():
    img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    labels = np.zeros((10, 10), dtype=np.int64)
    labels[1:3, 1:3] = 1
    labels[6:9, 6:9] = 2
    output = label2rgb(labels, image=img, bg_label=0)
    # Make sure that the output is made of floats and in the correct range
    assert np.issubdtype(output.dtype, np.floating)
    assert output.max() <= 1


def test_rgb():
    image = np.ones((1, 3))
    label = np.arange(3).reshape(1, -1)
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # Set alphas just in case the defaults change
    rgb = label2rgb(label, image=image, colors=colors, alpha=1,
                    image_alpha=1, bg_label=-1)
    assert_array_almost_equal(rgb, [colors])


def test_alpha():
    image = np.random.uniform(size=(3, 3))
    label = np.random.randint(0, 9, size=(3, 3))
    # If we set `alpha = 0`, then rgb should match image exactly.
    rgb = label2rgb(label, image=image, alpha=0, image_alpha=1,
                    bg_label=-1)
    assert_array_almost_equal(rgb[..., 0], image)
    assert_array_almost_equal(rgb[..., 1], image)
    assert_array_almost_equal(rgb[..., 2], image)


def test_no_input_image():
    label = np.arange(3).reshape(1, -1)
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    rgb = label2rgb(label, colors=colors, bg_label=-1)
    assert_array_almost_equal(rgb, [colors])


def test_image_alpha():
    image = np.random.uniform(size=(1, 3))
    label = np.arange(3).reshape(1, -1)
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # If we set `image_alpha = 0`, then rgb should match label colors exactly.
    rgb = label2rgb(label, image=image, colors=colors, alpha=1,
                    image_alpha=0, bg_label=-1)
    assert_array_almost_equal(rgb, [colors])


def test_color_names():
    image = np.ones((1, 3))
    label = np.arange(3).reshape(1, -1)
    cnames = ['red', 'lime', 'blue']
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # Set alphas just in case the defaults change
    rgb = label2rgb(label, image=image, colors=cnames, alpha=1,
                    image_alpha=1, bg_label=-1)
    assert_array_almost_equal(rgb, [colors])


def test_bg_and_color_cycle():
    image = np.zeros((1, 10))  # dummy image
    label = np.arange(10).reshape(1, -1)
    colors = [(1, 0, 0), (0, 0, 1)]
    bg_color = (0, 0, 0)
    rgb = label2rgb(label, image=image, bg_label=0, bg_color=bg_color,
                    colors=colors, alpha=1)
    assert_array_almost_equal(rgb[0, 0], bg_color)
    for pixel, color in zip(rgb[0, 1:], itertools.cycle(colors)):
        assert_array_almost_equal(pixel, color)


def test_negative_labels():
    labels = np.array([0, -1, -2, 0])
    rout = np.array([(0., 0., 0.), (0., 0., 1.), (1., 0., 0.), (0., 0., 0.)])
    assert_array_almost_equal(
        rout, label2rgb(labels, bg_label=0, alpha=1, image_alpha=1))


def test_nonconsecutive():
    labels = np.array([0, 2, 4, 0])
    colors = [(1, 0, 0), (0, 0, 1)]
    rout = np.array([(1., 0., 0.), (0., 0., 1.), (1., 0., 0.), (1., 0., 0.)])
    assert_array_almost_equal(
        rout, label2rgb(labels, colors=colors, alpha=1,
                        image_alpha=1, bg_label=-1))


def test_label_consistency():
    """Assert that the same labels map to the same colors."""
    label_1 = np.arange(5).reshape(1, -1)
    label_2 = np.array([0, 1])
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)]
    # Set alphas just in case the defaults change
    rgb_1 = label2rgb(label_1, colors=colors, bg_label=-1)
    rgb_2 = label2rgb(label_2, colors=colors, bg_label=-1)
    for label_id in label_2.flat:
        assert_array_almost_equal(rgb_1[label_1 == label_id],
                                  rgb_2[label_2 == label_id])


def test_leave_labels_alone():
    labels = np.array([-1, 0, 1])
    labels_saved = labels.copy()

    label2rgb(labels, bg_label=-1)
    label2rgb(labels, bg_label=1)
    assert_array_equal(labels, labels_saved)


def test_avg():
    # label image
    label_field = np.array([[1, 1, 1, 2],
                            [1, 2, 2, 2],
                            [3, 3, 4, 4]], dtype=np.uint8)

    # color image
    r = np.array([[1., 1., 0., 0.],
                  [0., 0., 1., 1.],
                  [0., 0., 0., 0.]])
    g = np.array([[0., 0., 0., 1.],
                  [1., 1., 1., 0.],
                  [0., 0., 0., 0.]])
    b = np.array([[0., 0., 0., 1.],
                  [0., 1., 1., 1.],
                  [0., 0., 1., 1.]])
    image = np.dstack((r, g, b))

    # reference label-colored image
    rout = np.array([[0.5, 0.5, 0.5, 0.5],
                     [0.5, 0.5, 0.5, 0.5],
                     [0. , 0. , 0. , 0. ]])
    gout = np.array([[0.25, 0.25, 0.25, 0.75],
                     [0.25, 0.75, 0.75, 0.75],
                     [0.  , 0.  , 0.  , 0.  ]])
    bout = np.array([[0. , 0. , 0. , 1. ],
                     [0. , 1. , 1. , 1. ],
                     [0.0, 0.0, 1.0, 1.0]])
    expected_out = np.dstack((rout, gout, bout))

    # test standard averaging
    out = label2rgb(label_field, image, kind='avg', bg_label=-1)
    assert_array_equal(out, expected_out)

    # test averaging with custom background value
    out_bg = label2rgb(label_field, image, bg_label=2, bg_color=(0, 0, 0),
                       kind='avg')
    expected_out_bg = expected_out.copy()
    expected_out_bg[label_field == 2] = 0
    assert_array_equal(out_bg, expected_out_bg)

    # test default background color
    out_bg = label2rgb(label_field, image, bg_label=2, kind='avg')
    assert_array_equal(out_bg, expected_out_bg)


def test_negative_intensity():
    labels = np.arange(100).reshape(10, 10)
    image = np.full((10, 10), -1, dtype='float64')
    assert_warns(UserWarning, label2rgb, labels, image, bg_label=-1)


def test_bg_color_rgb_string():
    img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    labels = np.zeros((10, 10), dtype=np.int64)
    labels[1:3, 1:3] = 1
    labels[6:9, 6:9] = 2
    output = label2rgb(labels, image=img, alpha=0.9, bg_label=0, bg_color='red')
    assert output[0, 0, 0] > 0.9 # red channel


def test_avg_with_2d_image():
    img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    labels = np.zeros((10, 10), dtype=np.int64)
    labels[1:3, 1:3] = 1
    labels[6:9, 6:9] = 2
    assert_no_warnings(label2rgb, labels, image=img, bg_label=0, kind='avg')
