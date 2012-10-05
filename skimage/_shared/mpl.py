import numpy as np
from matplotlib.colors import LinearSegmentedColormap


__all__ = ['LinearColormap', 'ClearColormap']


class LinearColormap(LinearSegmentedColormap):
    """Create Matplotlib colormap with color values specified at key points.

    This class simplifies the call signature of LinearSegmentedColormap.
    Colors specified by `color_data` are equally spaced along the colormap.

    Parameters
    ----------
    name : str
        Name of colormap.
    color_data : list
        Colors at each index value. List of RGB or RGBA tuples. For example,
        red and blue::

                color_data = [(1, 0, 0), (0, 0, 1)]
    """

    def __init__(self, name, color_data, index=None, **kwargs):
        color_data = rgb_list_to_colordict(color_data)
        index = np.linspace(0, 1, len(color_data['red']))
        # Adapt color_data to the form expected by LinearSegmentedColormap.
        color_data = dict((key, [(x, y, y) for x, y in zip(index, value)])
                          for key, value in color_data.iteritems())
        LinearSegmentedColormap.__init__(self, name, color_data, **kwargs)


def rgb_list_to_colordict(rgb_list):
    colors_by_channel = zip(*rgb_list)
    channels = ('red', 'green', 'blue', 'alpha')
    return dict((color, value)
                for color, value in zip(channels, colors_by_channel))


class ClearColormap(LinearColormap):
    """Color map that varies linearly from alpha = 0 to 1
    """
    def __init__(self, rgb, max_alpha=1, name='clear_color'):
        color_data = [tuple(rgb) + (0,), tuple(rgb) + (max_alpha,)]
        LinearColormap.__init__(self, name, color_data)
