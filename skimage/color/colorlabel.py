import os
import ast
import warnings
import itertools
import ConfigParser

import numpy as np

from skimage import img_as_float
from skimage._shared.utils import is_str
from .colorconv import rgb2gray, gray2rgb


__all__ = ['color_dict', 'image_label2rgb', 'DEFAULT_COLORS']


DEFAULT_COLORS = ('red', 'blue', 'yellow', 'magenta', 'green',
                  'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen')


def _load_colors():
    directory, _ = os.path.split(os.path.abspath(__file__))
    config_file = os.path.join(directory, 'colors.ini')

    parser = ConfigParser.ConfigParser()
    parser.read(config_file)
    return dict((k, ast.literal_eval(v)) for k, v in parser.items('colors'))
color_dict = _load_colors()


def _rgb_vector(color):
    """Return RGB color as (1, 3) array.

    This RGB array gets multiplied by masked regions of an RGB image, which are
    partially flattened by masking (i.e. dimensions 2D + RGB -> 1D + RGB).

    Parameters
    ----------
    color : str or array
        Color name in `color_dict` or RGB float values between [0, 1].
    """
    if is_str(color):
        color = color_dict[color]
    # slice to handle RGBA colors
    return np.array(color[:3]).reshape(1, 3)


def image_label2rgb(image, label, colors=None, alpha=0.3,
                    bg_label=-1, bg_color=None, image_alpha=1):
    """Return an RGB image where color-coded labels are painted over the image.

    Parameters
    ----------
    image : array
        Input image. If the input is an RGB image, it's converted to grayscale
        before coloring.
    label : array
        Integer array of labels with the same shape as `image`.
    colors : list
        List of colors. If the number of labels exceeds the number of colors,
        then the colors are cycled.
    alpha : float [0, 1]
        Opacity of colorized labels.
    bg_label : int
        Label that's treated as the background.
    bg_color : str or array
        Background color. Must be a name in `color_dict` or RGB float values
        between [0, 1].
    image_alpha : float [0, 1]
        Opacity of the image.
    """
    if not image.shape[:2] == label.shape:
        raise ValueError("`image` and `label` must be the same shape")

    if image.min() < 0:
        warnings.warn("Negative intensities in `image` are not supported")

    if colors is None:
        colors = DEFAULT_COLORS
    colors = [_rgb_vector(c) for c in colors]

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

    for c, i in itertools.izip(color_cycle, labels):
        mask = (label == i)
        colorized[mask] = c * alpha + colorized[mask] * (1 - alpha)

    return colorized
