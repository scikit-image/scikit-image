import warnings
import itertools

import numpy as np

from skimage import img_as_float
from skimage._shared import six
from skimage._shared.six.moves import zip
from .colorconv import rgb2gray, gray2rgb
from . import rgb_colors


__all__ = ['color_dict', 'label2rgb', 'DEFAULT_COLORS']


DEFAULT_COLORS = ('red', 'blue', 'yellow', 'magenta', 'green',
                  'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen')


color_dict = rgb_colors.__dict__


def _rgb_vector(color):
    """Return RGB color as (1, 3) array.

    This RGB array gets multiplied by masked regions of an RGB image, which are
    partially flattened by masking (i.e. dimensions 2D + RGB -> 1D + RGB).

    Parameters
    ----------
    color : str or array
        Color name in `color_dict` or RGB float values between [0, 1].
    """
    if isinstance(color, six.string_types):
        color = color_dict[color]
    # slice to handle RGBA colors
    return np.array(color[:3]).reshape(1, 3)


def label2rgb(label, image=None, colors=None, alpha=0.3,
              bg_label=-1, bg_color=None, image_alpha=1):
    """Return an RGB image where color-coded labels are painted over the image.

    Parameters
    ----------
    label : array
        Integer array of labels with the same shape as `image`.
    image : array
        Image used as underlay for labels. If the input is an RGB image, it's
        converted to grayscale before coloring.
    colors : list
        List of colors. If the number of labels exceeds the number of colors,
        then the colors are cycled.
    alpha : float [0, 1]
        Opacity of colorized labels. Ignored if image is `None`.
    bg_label : int
        Label that's treated as the background.
    bg_color : str or array
        Background color. Must be a name in `color_dict` or RGB float values
        between [0, 1].
    image_alpha : float [0, 1]
        Opacity of the image.
    """
    if colors is None:
        colors = DEFAULT_COLORS
    colors = [_rgb_vector(c) for c in colors]

    if image is None:
        colorized = np.zeros(label.shape + (3,), dtype=np.float64)
        # Opacity doesn't make sense if no image exists.
        alpha = 1
    else:
        if not image.shape[:2] == label.shape:
            raise ValueError("`image` and `label` must be the same shape")

        if image.min() < 0:
            warnings.warn("Negative intensities in `image` are not supported")

        image = img_as_float(rgb2gray(image))
        colorized = gray2rgb(image) * image_alpha + (1 - image_alpha)

    labels = list(set(label.flat))
    color_cycle = itertools.cycle(colors)

    if bg_label in labels:
        labels.remove(bg_label)
        if bg_color is not None:
            labels.insert(0, bg_label)
            bg_color = _rgb_vector(bg_color)
            color_cycle = itertools.chain(bg_color, color_cycle)

    for c, i in zip(color_cycle, labels):
        mask = (label == i)
        colorized[mask] = c * alpha + colorized[mask] * (1 - alpha)

    return colorized
