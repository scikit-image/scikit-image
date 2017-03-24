import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def manual(image):
    """Return a binary image based on the selections made with mouse clicks.

    Parameters
    ----------
    image : (M, 2) array
        Grayscale or RGB 2D image.

    Returns
    -------
    mask : (M, 2) binary array
        Segmented binary mask.

    Notes
    -----
    Use the cursor to draw objects.

    Examples
    --------
    >>> from skimage import data, segmentation
    >>> camera = data.camera()
    >>> mask = segmentation.manual(camera)
    >>> # Use the cursor to draw objects
    >>> io.imshow(mask)
    >>> io.show()

    """

    image = np.squeeze(image)

    if image.ndim is not 2:
        raise TypeError('Only 2-D images supported.')

    manual.list_of_verts = []

    _select_lasso(image)

    mask = np.zeros_like(image)

    yshape, xshape = image.shape
    y_grid, x_grid = np.mgrid[:yshape, :xshape]
    all_pixels = np.vstack((x_grid.ravel(), y_grid.ravel())).T

    for verts in manual.list_of_verts:
        _path = matplotlib.path.Path(verts)
        _mask = _path.contains_points(all_pixels)
        _mask = _mask.reshape(image.shape)
        mask += _mask

    return mask


def _select_lasso(image):
    """Uses the LassoSelector widget from matplotlib
    to crop freehand."""
    fig, ax = plt.subplots()
    ax.imshow(image)
    lasso = matplotlib.widgets.LassoSelector(ax, _onselect)
    plt.show()


def _onselect(verts):
    manual.list_of_verts.append(verts)
    plt.plot(*zip(*verts), linewidth=1)
    plt.draw()
