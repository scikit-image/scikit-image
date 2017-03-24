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

    if image.ndim not in (2, 3):
        raise TypeError('Only 2-D images or 3-D images supported.')

    manual.list_of_verts = []
    manual.line_selections = []

    _select_lasso(image)

    if image.ndim is 3:
        image = np.squeeze(image[:, :, :1])
    mask = np.zeros_like(image)

    yshape, xshape = image.shape
    y_grid, x_grid = np.mgrid[:yshape, :xshape]
    all_pixels = np.vstack((x_grid.ravel(), y_grid.ravel())).T

    for verts in manual.list_of_verts:
        _path = matplotlib.path.Path(verts, closed=True)
        _mask = _path.contains_points(all_pixels)
        _mask = _mask.reshape(image.shape)
        mask += _mask

    return mask >= 1


def _select_lasso(image):
    """Uses the LassoSelector widget from matplotlib
    to crop freehand."""
    fig, _select_lasso.ax = plt.subplots()
    _select_lasso.ax.imshow(image)

    buttonpos = plt.axes([0.85, 0.5, 0.1, 0.075])
    undo_button = matplotlib.widgets.Button(buttonpos, "Undo")
    undo_button.on_clicked(_undo)

    lasso = matplotlib.widgets.LassoSelector(_select_lasso.ax, _onselect)
    plt.show()


def _onselect(verts):
    manual.list_of_verts.append(verts)
    previous, = _select_lasso.ax.plot(*zip(*verts), linewidth=1)
    manual.line_selections.append(previous)
    plt.draw()


def _undo(event):
    if len(manual.list_of_verts) > 0:
        del manual.list_of_verts[-1]
        manual.line_selections[-1].remove()
        del manual.line_selections[-1]
    else:
        pass
