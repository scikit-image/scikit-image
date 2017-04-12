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


def manual(image, alpha=0.4, overlap="merge", mouse=[1, 2, 3]):
    """Return a binary image based on the selections made with mouse clicks.

    Parameters
    ----------
    image : (M, N[, 3]) array
        Grayscale or RGB image.

    alpha : float, optional
        Transparancy value for polygons draw over the segments.

    overlap : string, optional
        Determines the behaviour of mask generation with respect to overlapping
        regions of polygons.

        * if "merge" : overlapping selections will be merged as a single
            object.
        * if "overwrite" : The last polygon overwrites the regions where
            it intersects with previously drawn polygon.
        * if "seperate" : overlapping selections will be returned as
            an array of masks.

    mouse : list of int, optional
        Customize the mouse click behaviour when selecting vertices with
        the polygonal mode. [left, middle, right] is the default behaviour.

        * `1` : select a vertex.
        * '2' : undo previously selected vertex.
        * '3' : confirm selected polygon.

        To use right click to undo previously selected vertex, and middle
        button to confirm selection, pass `mouse=[1,3,2]` to the function.


    Returns
    -------
    mask : array or list of arrays

        * if `overlap` is "merge" : (M, N) boolean image with segmented
          regions.
        * if `overlap` is "overwrite" : (M, N) labelled image with segmented
          regions.
        * if `overlap` is "seperate" : array of (M, N) boolean images.

    Notes
    -----
    Use the cursor to mark objects in the image.

    In the lasso mode, press and hold the left mouse button and draw around
    the object.

    In the polygon mode, use left click to select the vertices of the polygon
    and right click to confirm the selection once the vertices are selected.
    Middle click will undo the previously selected vertex. This behaviour can
    be customized by with the optional 'mouse' parameter.

    By default, overlapping segments are merged into a single object.
    The optional paramter 'return_masks if 'True' will return an array of
    masks with a single selection per array.

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
    patch_objects = []

    temp_list = []
    preview_polygon_drawn = []

    left, middle, right = mouse

    def _draw_polygon(verts, alpha=alpha):
        polygon = Polygon(verts, closed=True)
        p = PatchCollection(
            [polygon], match_original=True, cmap=plt.cm.spectral, alpha=alpha)
        polygon_object = ax.add_collection(p)
        plt.draw()
        return polygon_object

    def _draw_lines(event):
        # Do not record click events from pressing the buttons.
        if (event.inaxes is lasso_pos or
                event.inaxes is polygon_pos or
                event.inaxes is undo_pos or
                event.inaxes is clear_pos):
            return

        # Do not record click events when toolbar is active.
        elif fig.canvas.manager.toolbar._active is not None:
            return

        # Do not record click events outside axis.
        elif event.inaxes is None:
            return

        elif event.button == left:  # Select vertex
            temp_list.append([event.xdata, event.ydata])
            # Remove previously drawn preview polygon if any.
            if len(preview_polygon_drawn) > 0:
                preview_polygon_drawn[-1].remove()
                preview_polygon_drawn.remove(preview_polygon_drawn[-1])

            # Preview polygon with selected vertices.
            polygon = _draw_polygon(temp_list, alpha=alpha/1.4)
            preview_polygon_drawn.append(polygon)

        elif event.button == middle:  # Undo previously selected vertex
            if len(temp_list) > 1:

                # Remove the previous vertice and update preview polygon
                temp_list.remove(temp_list[-1])
                preview_polygon_drawn[-1].remove()
                preview_polygon_drawn.remove(preview_polygon_drawn[-1])

                polygon = _draw_polygon(temp_list, alpha=alpha/1.4)
                preview_polygon_drawn.append(polygon)

            else:
                return

        elif event.button == right:  # Confirm the selection
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
        if len(verts) < 3:
            return
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

    def _undo_all(event):
        for _ in range(len(list_of_verts)):
            _undo(None)

    def _mode_lasso(event):
        global cid

        plt.suptitle("Draw around an object to select for mask generation.")

        # Disconnects previous lasso objects.
        # This is useful when switching between segmentation modes.
        if len(lasso_objects) > 0:
            lasso_objects[-1].disconnect_events()
            lasso_objects.remove(lasso_objects[-1])

        lasso = matplotlib.widgets.LassoSelector(ax, _on_select)
        lasso_objects.append(lasso)

        # When switching from polygonal selection mode.
        fig.canvas.mpl_disconnect(cid)
        plt.show(block=True)

    def _mode_polygonal(event):
        global cid
        if mouse is [1, 2, 3]:
            plt.suptitle(
                "Left click to select vertex. Right click to confirm vertices.\
            \nMiddle click to undo previous vertex.")

        # When switching from lasso selection mode.
        if len(lasso_objects) > 0:
            lasso_objects[-1].disconnect_events()
            lasso_objects.remove(lasso_objects[-1])

        cid = fig.canvas.mpl_connect('button_press_event', _draw_lines)

    global cid
    cid = None

    lasso_objects = []

    image = np.squeeze(image)

    if image.ndim not in (2, 3):
        raise ValueError('Only 2-D images or 3-D images supported.')

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.imshow(image, cmap="gray")
    ax.set_axis_off()

    # Buttons
    lasso_pos = plt.axes([0.85, 0.75, 0.10, 0.075])
    lasso_button = matplotlib.widgets.Button(lasso_pos, "Lasso")
    lasso_button.on_clicked(_mode_lasso)

    polygon_pos = plt.axes([0.85, 0.65, 0.10, 0.075])
    line_button = matplotlib.widgets.Button(polygon_pos, "Polygon")
    line_button.on_clicked(_mode_polygonal)

    undo_pos = plt.axes([0.59, 0.05, 0.075, 0.075])
    undo_button = matplotlib.widgets.Button(undo_pos, u'\u27F2')
    undo_button.on_clicked(_undo)

    clear_pos = plt.axes([0.675, 0.05, 0.10, 0.075])
    clear_button = matplotlib.widgets.Button(clear_pos, "Clear All")
    clear_button.on_clicked(_undo_all)

    # Default mode
    default = _mode_lasso(None)
    plt.show(block=True)

    if "seperate" in overlap:
        mask = np.array(
            [_mask_from_verts(verts,
                              image.shape[:2]) for verts in list_of_verts])
        return mask

    mask = np.zeros(image.shape[:2])

    for i, verts in enumerate(list_of_verts, start=1):
        cur_mask = _mask_from_verts(verts, image.shape[:2])
        mask[cur_mask] = i

    if "overwrite" in overlap:
        return mask

    elif "merge" in overlap:
        return mask > 0
