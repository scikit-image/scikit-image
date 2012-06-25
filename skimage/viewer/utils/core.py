import numpy as np
import matplotlib.pyplot as plt


__all__ = ['figimage', 'toolbar_off']


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


class toolbar_off(object):
    """Context manager to remove toolbar from a figure

    This is a terrible hack, but I couldn't figure out a GUI-neutral way to
    remove toolbars.

    Examples
    --------
    >>> with toolbar_off():
    ...     plt.figure()

    """

    def __init__(self, no_toolbar=True):
        self.no_toolbar = no_toolbar

    def __enter__(self):
        self.original_state = plt.rcParams['toolbar']
        if self.no_toolbar:
            plt.rcParams['toolbar'] = 'none'

    def __exit__(self, type, value, traceback):
        plt.rcParams['toolbar'] = self.original_state

