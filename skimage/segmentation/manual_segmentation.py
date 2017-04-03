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


def manual(image, alpha=0.4, return_masks=False, draw_lines=False):
    """Return a binary image based on the selections made with mouse clicks.

    Parameters
    ----------
    image : (M, N[, 3]) array
        Grayscale or RGB image.

    alpha : float, optional
        Transparancy value for polygons draw over the segments.

    return_masks : bool, optional
        If True, return a list of mask with individual selections.

    draw_lines : bool, optional
        If True, select atleast three points to define the edges
        of a polygon.

        * Left click to select a point as vertice.
        * Middle click to undo previously selected vertice.
        * Right click to confirm the selected points as vertices of a polygon.

    Returns
    -------
    mask : array or list of arrays

        * if `return_masks` is True : list of (M, N) boolean images.
        * if `return_masks` is False: (M, N) boolean image with segmented
          regions.

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

    temp_list = []
    preview_polygon_drawn = []

    def _draw_polygon(verts, alpha=alpha):
        polygon = Polygon(verts, closed=True)
        p = PatchCollection(
            [polygon], match_original=True, alpha=alpha)
        polygon_object = ax.add_collection(p)
        plt.draw()
        return polygon_object

    def _draw_lines(event):
        # Do not record click events from pressing the Undo button.
        if event.inaxes is buttonpos:
            return

        # Do not record click events when toolbar is active.
        if fig.canvas.manager.toolbar._active is not None:
            return

        # Do not record click events outside axis.
        if event.inaxes is None:
            return

        elif event.button == 1:  # Left click
            temp_list.append([event.xdata, event.ydata])

            # Remove previously drawn preview polygon if any.
            if len(preview_polygon_drawn) > 0:
                preview_polygon_drawn[-1].remove()
                preview_polygon_drawn.remove(preview_polygon_drawn[-1])

            # Preview polygon with selected vertices.
            polygon = _draw_polygon(temp_list, alpha=alpha/1.2)
            preview_polygon_drawn.append(polygon)

        elif event.button == 2:  # Middle click
            if len(temp_list) > 1:

                # Remove the previous vertice and update preview polygon
                temp_list.remove(temp_list[-1])
                preview_polygon_drawn[-1].remove()
                preview_polygon_drawn.remove(preview_polygon_drawn[-1])

                polygon = _draw_polygon(temp_list, alpha=alpha/1.2)
                preview_polygon_drawn.append(polygon)

            else:
                return

        elif event.button == 3:  # Right click
            if len(temp_list) is 0:
                return

            # Store the vertices of the polygon as shown in preview.
            # Redraw polygon and store it in polygons_drawn so that
            # `_undo` works correctly.
            list_of_verts.append(list(temp_list))
            polygon_object = _draw_polygon(list_of_verts[-1])
            polygons_drawn.append(polygon_object)

            # Empty the temporary variables.
            preview_polygon_drawn[-1].remove()
            preview_polygon_drawn.remove(preview_polygon_drawn[-1])
            del temp_list[:]

            plt.draw()

    def _on_select(verts):
        list_of_verts.append(verts)
        polygon_object = _draw_polygon(verts)
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
            return

    image = np.squeeze(image)

    if image.ndim not in (2, 3):
        raise TypeError('Only 2-D images or 3-D images supported.')

    fig, ax = plt.subplots()
    ax.imshow(image)

    buttonpos = plt.axes([0.85, 0.45, 0.075, 0.075])
    undo_button = matplotlib.widgets.Button(buttonpos, u'\u27F2')

    undo_button.on_clicked(_undo)

    if draw_lines is True:
        plt.suptitle(
            "Left click to select vertex. Right click to confirm vertices.\
        \nMiddle click to undo previous vertex.")

        cid = fig.canvas.mpl_connect('button_press_event', _draw_lines)

    else:
        plt.suptitle("Draw around an object to select for mask generation.")
        lasso = matplotlib.widgets.LassoSelector(ax, _on_select)

    plt.show(block=True)

    if return_masks is False:
        mask = np.zeros(image.shape[:2], dtype=bool)
        for verts in list_of_verts:
            mask += _mask_from_verts(verts, image.shape[:2])
    else:
        mask = np.array(
            [_mask_from_verts(verts,
                              image.shape[:2]) for verts in list_of_verts])
    return mask >= 1
