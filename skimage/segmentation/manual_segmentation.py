import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def manual(image, speed_up=1):
    """Return a binary image based on the selections made with mouse clicks.

    Parameters
    ----------
    image : (M, 2) array
        Grayscale or RGB 2D image.

    speed_up : int
        Skips vertices in integer steps to speed up mask generation. This
        must be a non-zero integer.

    Returns
    -------
    mask : (M, 2) binary array
        Segmented binary mask.

    Notes
    -----
    Use the cursor to draw objects. Increasing speed_up value will 
    generate the mask faster, but the objects will be less smoother. 

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
    manual.polygons_selection = []
    manual.polygon_list = []

    _select_lasso(image)

    if image.ndim is 3:
        image = np.squeeze(image[:, :, :1])

    mask = np.zeros(image.shape)

    yshape, xshape = image.shape
    y_grid, x_grid = np.mgrid[:yshape, :xshape]
    all_pixels = np.vstack((x_grid.ravel(), y_grid.ravel())).T

    for verts in manual.list_of_verts:
        verts_ = verts[::speed_up]
        _path = matplotlib.path.Path(verts_, closed=True)
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

    polygon = Polygon(verts, True)
    manual.polygon_list.append(polygon)

    p = PatchCollection(manual.polygon_list, alpha=0.4)
    polygon_object = _select_lasso.ax.add_collection(p)

    manual.polygons_selection.append(polygon_object)

    plt.draw()


def _undo(event):
    if len(manual.list_of_verts) > 0:
        manual.list_of_verts.remove(manual.list_of_verts[-1])
        manual.polygons_selection[-1].remove()
        manual.polygons_selection.remove(manual.polygons_selection[-1])
        manual.polygon_list.remove(manual.polygon_list[-1])
    else:
        pass
