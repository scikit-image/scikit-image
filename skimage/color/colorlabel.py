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
        img_layer = np.zeros(label.shape + (3,), dtype=np.float64)
        # Opacity doesn't make sense if no image exists.
        alpha = 1
    else:
        if not image.shape[:2] == label.shape:
            raise ValueError("`image` and `label` must be the same shape")

        if image.min() < 0:
            warnings.warn("Negative intensities in `image` are not supported")

        image = img_as_float(rgb2gray(image))
        img_layer = gray2rgb(image) * image_alpha + (1 - image_alpha)

    # need to ensure that all labels are ints >= 0
    offset = label.min()
    if offset != 0:
        label -= offset
        bg_label -= offset
    new_type = np.min_scalar_type(label.max())
    if new_type == np.bool:
        new_type = np.uint8
    label = label.astype(new_type)

    labels = list(set(label.flat))
    color_cycle = itertools.cycle(colors)

    remove_background = bg_label in labels and bg_color is None

    if bg_label in labels:
        labels.remove(bg_label)
        if bg_color is not None:
            labels.insert(0, bg_label)
            bg_color = _rgb_vector(bg_color)
            color_cycle = itertools.chain(bg_color, color_cycle)

    if len(labels) == 0:
        return img_layer

    label_to_color = np.zeros((max(labels) + 1, 3))
    for lab, c in zip(labels, color_cycle):
        label_to_color[lab] = c

    label_layer = label_to_color[label]
    result = label_layer * alpha + img_layer * (1 - alpha)

    # remove background label if its color was not specified
    if remove_background:
        result[label == bg_label] = img_layer[label == bg_label]

    return result
