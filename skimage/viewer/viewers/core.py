import sys

from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg

from skimage.viewer.utils import figimage


qApp = None


class ImageCanvas(FigureCanvasQTAgg):
    """Canvas for displaying images."""
    def __init__(self, parent, image, **kwargs):
        self.fig, self.ax = figimage(image, **kwargs)

        FigureCanvasQTAgg.__init__(self, self.fig)
        FigureCanvasQTAgg.setSizePolicy(self,
                                        QtGui.QSizePolicy.Expanding,
                                        QtGui.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)
        # Note: `setParent` must be called after `FigureCanvasQTAgg.__init__`.
        self.setParent(parent)


class ImageViewer(QtGui.QMainWindow):
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
        Plugins typically operate on (but don't change) the original image.
    plugins : list
        List of attached plugins.
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
        self.image = image
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

        self.connect_event('motion_notify_event', self.update_status_bar)

    def __iadd__(self, plugin):
        plugin.attach(self)
        return self

    def closeEvent(self, ce):
        self.close()

    def auto_layout(self):
        """Move viewer to top-left and align plugin on right edge of viewer."""
        size = self.geometry()
        self.move(0, 0)
        w = size.width()
        y = 0
        #TODO: Layout isn't correct for multiple plugins (overlaps).
        for p in self.plugins:
            p.move(w, y)
            y += p.geometry().height()

    def show(self):
        self.auto_layout()
        for p in self.plugins:
            p.show()
        super(ImageViewer, self).show()
        sys.exit(qApp.exec_())

    def redraw(self):
        self.canvas.draw_idle()

    @property
    def image(self):
        return self._img

    @image.setter
    def image(self, image):
        self._img = image
        self._image_plot.set_array(image)
        self.redraw()

    def connect_event(self, event, callback):
        """Connect callback function to matplotlib event and return id."""
        cid = self.canvas.mpl_connect(event, callback)
        return cid

    def disconnect_event(self, callback_id):
        """Disconnect callback by its id (returned by `connect_event`)."""
        self.canvas.mpl_disconnect(callback_id)

    def add_artist(self, artist):
        """Add matplotlib artist to image viewer."""
        self.ax.add_artist(artist)

    def remove_artist(self, artist):
        """Disconnect matplotlib artist from image viewer."""
        # There's probably a smarter way to do this.
        for artist_list in self._axes_artists:
            if artist in artist_list:
                artist_list.remove(artist)

    def update_status_bar(self, event):
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
