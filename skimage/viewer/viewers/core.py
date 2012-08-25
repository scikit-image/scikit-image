"""
ImageViewer class for viewing and interacting with images.
"""
import sys

try:
    from PyQt4 import QtGui, QtCore
    from PyQt4.QtGui import QMainWindow
except ImportError:
    QMainWindow = object # hack to prevent nosetest and autodoc errors
    print("Could not import PyQt4 -- skimage.viewer not available.")

from skimage.util.dtype import dtype_range
from ..utils import figimage, MatplotlibCanvas


qApp = None


class ImageCanvas(MatplotlibCanvas):
    """Canvas for displaying images."""
    def __init__(self, parent, image, **kwargs):
        self.fig, self.ax = figimage(image, **kwargs)
        super(ImageCanvas, self).__init__(parent, self.fig, **kwargs)


class ImageViewer(QMainWindow):
    """Viewer for displaying images.

    This viewer is a simple container object that holds a Matplotlib axes
    for showing images. `ImageViewer` doesn't subclass the Matplotlib axes (or
    figure) because of the high probability of name collisions.

    Parameters
    ----------
    image : array
        Image being viewed.

    Attributes
    ----------
    canvas, fig, ax : Matplotlib canvas, figure, and axes
        Matplotlib canvas, figure, and axes used to display image.
    image : array
        Image being viewed. Setting this value will update the displayed frame.
    original_image : array
        Plugins typically operate on (but don't change) the *original* image.
    plugins : list
        List of attached plugins.

    Examples
    --------
    >>> viewer = ImageViewer(image)
    >>> viewer += SomePlugin()
    >>> viewer.show()

    """
    def __init__(self, image):
        # Start main loop
        global qApp
        if qApp is None:
            qApp = QtGui.QApplication(sys.argv)
        super(ImageViewer, self).__init__()

        #TODO: Add ImageViewer to skimage.io window manager

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Image Viewer")

        self.file_menu = QtGui.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.close,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.main_widget = QtGui.QWidget()
        self.setCentralWidget(self.main_widget)

        self.canvas = ImageCanvas(self.main_widget, image)
        self.fig = self.canvas.fig
        self.ax = self.canvas.ax
        self.ax.autoscale(enable=False)

        self._image_plot = self.ax.images[0]

        self.original_image = image
        self.image = image.copy()
        self.plugins = []

        # List of axes artists to check for removal.
        self._axes_artists = [self.ax.artists,
                              self.ax.collections,
                              self.ax.images,
                              self.ax.lines,
                              self.ax.patches,
                              self.ax.texts]

        self.layout = QtGui.QVBoxLayout(self.main_widget)
        self.layout.addWidget(self.canvas)

        status_bar = self.statusBar()
        self.status_message = status_bar.showMessage
        sb_size = status_bar.sizeHint()
        cs_size = self.canvas.sizeHint()
        self.resize(cs_size.width(), cs_size.height() + sb_size.height())

        self.connect_event('motion_notify_event', self._update_status_bar)

    def __add__(self, plugin):
        """Add plugin to ImageViewer"""
        plugin.attach(self)
        return self

    def closeEvent(self, event):
        self.close()

    def auto_layout(self):
        """Move viewer to top-left and align plugin on right edge of viewer."""
        size = self.geometry()
        self.move(0, 0)
        w = size.width()
        y = 0
        #TODO: Layout isn't quite correct for multiple plugins (overlaps).
        for p in self.plugins:
            p.move(w, y)
            y += p.geometry().height()

    def show(self):
        """Show ImageViewer and attached plugins.

        This behaves much like `matplotlib.pyplot.show` and `QWidget.show`.
        """
        self.auto_layout()
        for p in self.plugins:
            p.show()
        super(ImageViewer, self).show()
        qApp.exec_()

    def redraw(self):
        self.canvas.draw_idle()

    @property
    def image(self):
        return self._img

    @image.setter
    def image(self, image):
        self._img = image
        self._image_plot.set_array(image)
        clim = dtype_range[image.dtype.type]
        self._image_plot.set_clim(clim)
        self.redraw()

    def reset_image(self):
        self.image = self.original_image.copy()

    def connect_event(self, event, callback):
        """Connect callback function to matplotlib event and return id."""
        cid = self.canvas.mpl_connect(event, callback)
        return cid

    def disconnect_event(self, callback_id):
        """Disconnect callback by its id (returned by `connect_event`)."""
        self.canvas.mpl_disconnect(callback_id)

    def remove_artist(self, artist):
        """Disconnect matplotlib artist from image viewer.

        The `closeEvent` method of a Plugin should remove artists (Matplotlib
        lines, markers, etc.) from the viewer so that they aren't stranded.

        Parameters
        ----------
        artist : Matplotlib Artist
            Artists created by Matplotlib functions (e.g., `plot` returns list
            of `Line2D` artists) should be saved by the plugin for removal.
        """
        # Note: an `add_artist` method is unnecessary since Matplotlib

        # There's probably a smarter way to find where the artist is stored.
        for artist_list in self._axes_artists:
            if artist in artist_list:
                artist_list.remove(artist)

    def _update_status_bar(self, event):
        if event.inaxes and event.inaxes.get_navigate():
            self.status_message(self._format_coord(event.xdata, event.ydata))
        else:
            self.status_message('')

    def _format_coord(self, x, y):
        # callback function to format coordinate display in status bar
        x = int(x + 0.5)
        y = int(y + 0.5)
        try:
            return "%4s @ [%4s, %4s]" % (self.image[y, x], x, y)
        except IndexError:
            return ""
