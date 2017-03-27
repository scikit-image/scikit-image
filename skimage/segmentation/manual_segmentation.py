import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from ..draw import polygon


def _mask_from_verts(verts, shape):
    mask = np.zeros(shape, dtype=bool)
    pr = [y for x, y in verts]
    pc = [x for x, y in verts]
    rr, cc = polygon(pr, pc, shape)
    mask[rr, cc] = 1
    return mask


def manual(image, alpha=0.4, return_list=False):
    """Return a binary image based on the selections made with mouse clicks.

    Parameters
    ----------
    image : (M, N[, 3]) array
        Grayscale or RGB image.

    alpha : float or None (optional)
        Transparancy value for polygons draw over the segments.

    return_list : bool (optional)
        returns a list of maks with individual selections.

    Returns
    -------
    mask : (M, N) array
        Boolean image with segmented regions.

    Notes
    -----
    Use the cursor to draw objects.

    Examples
    --------
    >>> from skimage import data, segmentation, io
    >>> camera = data.camera()
    >>> mask = segmentation.manual(camera)  # doctest: +SKIP
    >>> io.imshow(mask)  # doctest: +SKIP
    >>> io.show()  # doctest: +SKIP

    """

    list_of_verts = []
    polygons_drawn = []

    def _on_select(verts):
        list_of_verts.append(verts)

        polygon = Polygon(verts, True)

        p = PatchCollection([polygon], match_original=True, alpha=alpha)
        polygon_object = ax.add_collection(p)
        polygons_drawn.append(polygon_object)
        plt.draw()

    def _undo(event):
        if len(list_of_verts) > 0:
            list_of_verts.remove(list_of_verts[-1])

            # Remove previous polygon object from the plot.
            polygons_drawn[-1].remove()

            # Removes latest polygon from a list of all polygon objects.
            # This enables undo till the first drawn polygon.
            polygons_drawn.remove(polygons_drawn[-1])

        else:
            pass

    image = np.squeeze(image)

    if image.ndim not in (2, 3):
        raise TypeError('Only 2-D images or 3-D images supported.')

    fig, ax = plt.subplots()
    ax.imshow(image)

    buttonpos = plt.axes([0.85, 0.45, 0.075, 0.075])
    undo_button = matplotlib.widgets.Button(buttonpos, u'\u27F2')
    undo_button.on_clicked(_undo)

    lasso = matplotlib.widgets.LassoSelector(ax, _on_select)
    plt.show(block=True)

    if return_list is False:
        mask = np.zeros(image.shape[:2], dtype=bool)
        for verts in list_of_verts:
            mask += _mask_from_verts(verts, image.shape[:2])
    else:
        mask = np.array(
            [_mask_from_verts(verts,
                              image.shape[:2]) for verts in list_of_verts])

    return mask >= 1
