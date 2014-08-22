from collections import namedtuple

import numpy as np
from numpy.testing import assert_equal
from numpy.testing.decorators import skipif
from skimage import data
from skimage.viewer import ImageViewer, viewer_available
from skimage.viewer.canvastools import (
    LineTool, ThickLineTool, RectangleTool, PaintTool)
from skimage.viewer.canvastools.base import CanvasToolBase
from numpy.testing import assert_equal
from numpy.testing.decorators import skipif


def get_end_points(image):
    h, w = image.shape[0:2]
    x = [w / 3, 2 * w / 3]
    y = [h / 2] * 2
    return np.transpose([x, y])


def create_mouse_event(ax, button=1, xdata=0, ydata=0, key=None):
    """
     *name*
        the event name

    *canvas*
        the FigureCanvas instance generating the event

    *guiEvent*
        the GUI event that triggered the matplotlib event

    *x*
        x position - pixels from left of canvas

    *y*
        y position - pixels from bottom of canvas

    *inaxes*
        the :class:`~matplotlib.axes.Axes` instance if mouse is over axes

    *xdata*
        x coord of mouse in data coords

    *ydata*
        y coord of mouse in data coords

     *button*
        button pressed None, 1, 2, 3, 'up', 'down' (up and down are used
        for scroll events)

    *key*
        the key depressed when the mouse event triggered (see
        :class:`KeyEvent`)

    *step*
        number of scroll steps (positive for 'up', negative for 'down')
    """
    event = namedtuple('Event',
                       ('name canvas guiEvent x y inaxes xdata ydata '
                        'button key step'))
    event.button = button
    event.x, event.y = ax.transData.transform((xdata, ydata))
    event.xdata, event.ydata = xdata, ydata
    event.inaxes = ax
    event.canvas = ax.figure.canvas
    event.key = key
    event.step = 1
    event.guiEvent = None
    event.name = 'Custom'
    return event


@skipif(not viewer_available)
def test_line_tool():
    img = data.camera()
    viewer = ImageViewer(img)

    tool = LineTool(viewer.ax, maxdist=10)
    tool.end_points = get_end_points(img)
    assert_equal(tool.end_points, np.array([[170, 256], [341, 256]]))

    # grab a handle and move it
    grab = create_mouse_event(viewer.ax, xdata=170, ydata=256)
    tool.on_mouse_press(grab)
    move = create_mouse_event(viewer.ax, xdata=180, ydata=260)
    tool.on_move(move)
    tool.on_mouse_release(move)
    assert_equal(tool.geometry, np.array([[180, 260], [341, 256]]))

    # create a new line
    new = create_mouse_event(viewer.ax, xdata=10, ydata=10)
    tool.on_mouse_press(new)
    move = create_mouse_event(viewer.ax, xdata=100, ydata=100)
    tool.on_move(move)
    tool.on_mouse_release(move)
    assert_equal(tool.geometry, np.array([[100, 100], [10, 10]]))


@skipif(not viewer_available)
def test_thick_line_tool():
    img = data.camera()
    viewer = ImageViewer(img)

    tool = ThickLineTool(viewer.ax, maxdist=10)
    tool.end_points = get_end_points(img)

    scroll_up = create_mouse_event(viewer.ax, button='up')
    tool.on_scroll(scroll_up)
    assert_equal(tool.linewidth, 2)

    scroll_down = create_mouse_event(viewer.ax, button='down')
    tool.on_scroll(scroll_down)
    assert_equal(tool.linewidth, 1)

    key_up = create_mouse_event(viewer.ax, key='+')
    tool.on_key_press(key_up)
    assert_equal(tool.linewidth, 2)

    key_down = create_mouse_event(viewer.ax, key='-')
    tool.on_key_press(key_down)
    assert_equal(tool.linewidth, 1)


@skipif(not viewer_available)
def test_rect_tool():
    img = data.camera()
    viewer = ImageViewer(img)

    tool = RectangleTool(viewer.ax, maxdist=10)
    tool.extents = (100, 150, 100, 150)

    assert_equal(tool.corners,
                 ((100, 150, 150, 100), (100, 100, 150, 150)))
    assert_equal(tool.extents, (100, 150, 100, 150))
    assert_equal(tool.edge_centers,
                 ((100, 125.0, 150, 125.0), (125.0, 100, 125.0, 150)))
    assert_equal(tool.geometry, (100, 150, 100, 150))

    # grab a corner and move it
    grab = create_mouse_event(viewer.ax, xdata=100, ydata=100)
    tool.press(grab)
    move = create_mouse_event(viewer.ax, xdata=120, ydata=120)
    tool.onmove(move)
    tool.release(move)
    assert_equal(tool.geometry, [120, 150, 120, 150])

    # create a new line
    new = create_mouse_event(viewer.ax, xdata=10, ydata=10)
    tool.press(new)
    move = create_mouse_event(viewer.ax, xdata=100, ydata=100)
    tool.onmove(move)
    tool.release(move)
    assert_equal(tool.geometry, [10, 100,  10, 100])


@skipif(not viewer_available)
def test_paint_tool():
    img = data.moon()
    viewer = ImageViewer(img)

    tool = PaintTool(viewer.ax, img.shape)

    tool.radius = 10
    assert_equal(tool.radius, 10)
    tool.label = 2
    assert_equal(tool.label, 2)
    assert_equal(tool.shape, img.shape)

    start = create_mouse_event(viewer.ax, xdata=100, ydata=100)
    tool.on_mouse_press(start)
    move = create_mouse_event(viewer.ax, xdata=110, ydata=110)
    tool.on_move(move)
    tool.on_mouse_release(move)
    assert_equal(tool.overlay[tool.overlay == 2].size, 761)

    tool.label = 5
    start = create_mouse_event(viewer.ax, xdata=20, ydata=20)
    tool.on_mouse_press(start)
    move = create_mouse_event(viewer.ax, xdata=40, ydata=40)
    tool.on_move(move)
    tool.on_mouse_release(move)
    assert_equal(tool.overlay[tool.overlay == 5].size, 881)
    assert_equal(tool.overlay[tool.overlay == 2].size, 761)

    enter = create_mouse_event(viewer.ax, key='enter')
    tool.on_mouse_press(enter)

    tool.overlay = tool.overlay * 0
    assert_equal(tool.overlay.sum(), 0)


@skipif(not viewer_available)
def test_base_tool():
    img = data.moon()
    viewer = ImageViewer(img)

    tool = CanvasToolBase(viewer.ax)
    tool.set_visible(False)
    tool.set_visible(True)

    enter = create_mouse_event(viewer.ax, key='enter')
    tool._on_key_press(enter)

    tool.redraw()
    tool.remove()

    tool = CanvasToolBase(viewer.ax, useblit=False)
    tool.redraw()
