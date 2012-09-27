"""
ImageViewer class for viewing and interacting with images.
"""
try:
    from PyQt4 import QtGui, QtCore
    from PyQt4.QtGui import QMainWindow
except ImportError:
    QMainWindow = object  # hack to prevent nosetest and autodoc errors
    print("Could not import PyQt4 -- skimage.viewer not available.")

from skimage.util.dtype import dtype_range
from .. import utils
from ..widgets import Slider


__all__ = ['ImageViewer', 'CollectionViewer']


class ImageCanvas(utils.MatplotlibCanvas):
    """Canvas for displaying images."""
    def __init__(self, parent, image, **kwargs):
        self.fig, self.ax = utils.figimage(image, **kwargs)
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
    >>> from skimage import data
    >>> image = data.coins()
    >>> viewer = ImageViewer(image)
    >>> # viewer.show()

    """
    def __init__(self, image):
        # Start main loop
        utils.init_qtapp()
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
        utils.start_qtapp()

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
        if clim[0] < 0 and image.min() >= 0:
            clim = (0, clim[1])
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


class CollectionViewer(ImageViewer):
    """Viewer for displaying image collections.

    Select the displayed frame of the image collection using the slider or
    with the following keyboard shortcuts:

        left/right arrows
            Previous/next image in collection.
        number keys, 0--9
            0% to 90% of collection. For example, "5" goes to the image in the
            middle (i.e. 50%) of the collection.
        home/end keys
            First/last image in collection.

    Subclasses and plugins will likely extend the `update_image` method to add
    custom overlays or filter the displayed image.

    Parameters
    ----------
    image_collection : list of images
        List of images to be displayed.
    update_on : {'on_slide' | 'on_release'}
        Control whether image is updated on slide or release of the image
        slider. Using 'on_release' will give smoother behavior when displaying
        large images or when writing a plugin/subclass that requires heavy
        computation.
    """

    def __init__(self, image_collection, update_on='move', **kwargs):
        self.image_collection = image_collection
        self.index = 0
        self.num_images = len(self.image_collection)

        first_image = image_collection[0]
        super(CollectionViewer, self).__init__(first_image)

        slider_kws = dict(value=0, low=0, high=self.num_images - 1)
        slider_kws['update_on'] = update_on
        slider_kws['callback'] = self.update_index
        slider_kws['value_type'] = 'int'
        self.slider = Slider('frame', **slider_kws)
        self.layout.addWidget(self.slider)

        #TODO: Adjust height to accomodate slider; the following doesn't work
        # s_size = self.slider.sizeHint()
        # cs_size = self.canvas.sizeHint()
        # self.resize(cs_size.width(), cs_size.height() + s_size.height())

    def update_index(self, name, index):
        """Select image on display using index into image collection."""
        index = int(round(index))

        if index == self.index:
            return

        # clip index value to collection limits
        index = max(index, 0)
        index = min(index, self.num_images - 1)

        self.index = index
        self.slider.val = index
        self.update_image(self.image_collection[index])

    def update_image(self, image):
        """Update displayed image.

        This method can be overridden or extended in subclasses and plugins to
        react to image changes.
        """
        self.image = image

    def keyPressEvent(self, event):
        if type(event) == QtGui.QKeyEvent:
            key = event.key()
            # Number keys (code: 0 = key 48, 9 = key 57) move to deciles
            if 48 <= key < 58:
                index = 0.1 * int(key - 48) * self.num_images
                self.update_index('', index)
            event.accept()
        else:
            event.ignore()
