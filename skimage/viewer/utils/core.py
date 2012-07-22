import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


__all__ = ['figimage', 'LinearColormap', 'ClearColormap']


def figimage(image, scale=1, dpi=None, **kwargs):
    """Return figure and axes with figure tightly surrounding image.

    Unlike pyplot.figimage, this actually plots onto an axes object, which
    fills the figure. Plotting the image onto an axes allows for subsequent
    overlays of axes artists.

    Parameters
    ----------
    image : array
        image to plot
    scale : float
        If scale is 1, the figure and axes have the same dimension as the
        image.  Smaller values of `scale` will shrink the figure.
    dpi : int
        Dots per inch for figure. If None, use the default rcParam.
    """
    dpi = dpi if dpi is not None else plt.rcParams['figure.dpi']
    kwargs.setdefault('interpolation', 'nearest')
    kwargs.setdefault('cmap', 'gray')

    h, w, d = np.atleast_3d(image).shape
    figsize = np.array((w, h), dtype=float) / dpi * scale

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

    ax.set_axis_off()
    ax.imshow(image, **kwargs)
    return fig, ax


class LinearColormap(LinearSegmentedColormap):
    """LinearSegmentedColormap in which color varies smoothly.

    This class is a simplification of LinearSegmentedColormap, which doesn't
    support jumps in color intensities.

    Parameters
    ----------
    name : str
        Name of colormap.

    segmented_data : dict
        Dictionary of 'red', 'green', 'blue', and (optionally) 'alpha' values.
        Each color key contains a list of `x`, `y` tuples. `x` must increase
        monotonically from 0 to 1 and corresponds to input values for a mappable
        object (e.g. an image). `y` corresponds to the color intensity.

    """
    def __init__(self, name, segmented_data, **kwargs):
        segmented_data = dict((key, [(x, y, y) for x, y in value])
                              for key, value in segmented_data.iteritems())
        LinearSegmentedColormap.__init__(self, name, segmented_data, **kwargs)


class ClearColormap(LinearColormap):
    """Color map that varies linearly from alpha = 0 to 1
    """
    def __init__(self, rgb, name='clear_color'):
        r, g, b = rgb
        cg_speq = {'blue':  [(0.0, b), (1.0, b)],
                   'green': [(0.0, g), (1.0, g)],
                   'red':   [(0.0, r), (1.0, r)],
                   'alpha': [(0.0, 0.0), (1.0, 1.0)]}
        LinearColormap.__init__(self, name, cg_speq)
