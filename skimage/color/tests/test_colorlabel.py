import itertools

import numpy as np
from numpy import testing
from skimage.color.colorlabel import image_label2rgb
from numpy.testing import assert_array_almost_equal as assert_close


def test_shape_mismatch():
    image = np.ones((3, 3))
    label = np.ones((2, 2))
    testing.assert_raises(ValueError, image_label2rgb, image, label)


def test_rgb():
    image = np.ones((1, 3))
    label = np.arange(3).reshape(1, -1)
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # Set alphas just in case the defaults change
    rgb = image_label2rgb(image, label, colors=colors, alpha=1, image_alpha=1)
    assert_close(rgb, [colors])


def test_alpha():
    image = np.random.uniform(size=(3, 3))
    label = np.random.randint(0, 9, size=(3, 3))
    # If we set `alpha = 0`, then rgb should match image exactly.
    rgb = image_label2rgb(image, label, alpha=0, image_alpha=1)
    assert_close(rgb[..., 0], image)
    assert_close(rgb[..., 1], image)
    assert_close(rgb[..., 2], image)


def test_image_alpha():
    image = np.random.uniform(size=(1, 3))
    label = np.arange(3).reshape(1, -1)
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # If we set `image_alpha = 0`, then rgb should match label colors exactly.
    rgb = image_label2rgb(image, label, colors=colors, alpha=1, image_alpha=0)
    assert_close(rgb, [colors])


def test_color_names():
    image = np.ones((1, 3))
    label = np.arange(3).reshape(1, -1)
    cnames = ['red', 'lime', 'blue']
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # Set alphas just in case the defaults change
    rgb = image_label2rgb(image, label, colors=cnames, alpha=1, image_alpha=1)
    assert_close(rgb, [colors])


def test_bg_and_color_cycle():
    image = np.zeros((1, 10))  # dummy image
    label = np.arange(10).reshape(1, -1)
    colors = [(1, 0, 0), (0, 0, 1)]
    bg_color = (0, 0, 0)
    rgb = image_label2rgb(image, label, bg_label=0, bg_color=bg_color,
                          colors=colors, alpha=1)
    assert_close(rgb[0, 0], bg_color)
    for pixel, color in zip(rgb[0, 1:], itertools.cycle(colors)):
        assert_close(pixel, color)


if __name__ == '__main__':
    testing.run_module_suite()

