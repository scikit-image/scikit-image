import matplotlib.pyplot as plt


def imshow(im, *args, **kwargs):
    """Show the input image and return the current axes.

    Parameters
    ----------
    im : array, shape (M, N[, 3])
        The image to display.

    *args, **kwargs : positional and keyword arguments
        These are passed directly to `matplotlib.pyplot.imshow`.

    Returns
    -------
    ax : `matplotlib.pyplot.Axes`
        The axes showing the image.
    """
    if plt.gca().has_data():
        plt.figure()
    kwargs.setdefault('interpolation', 'nearest')
    kwargs.setdefault('cmap', 'gray')
    return plt.imshow(im, *args, **kwargs)

imread = plt.imread
show = plt.show


def _app_show():
    show()
