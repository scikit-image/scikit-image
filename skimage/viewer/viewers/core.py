import sys

from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg

from skimage.viewer.utils import figimage, clear_red


qApp = None


class ImageCanvas(FigureCanvasQTAgg):
    """Canvas for displaying images.

    This canvas derives from Matplotlib, so your normal 
    """
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

    def __init__(self, image):
        # Start main loop
        global qApp
        if qApp is None:
            qApp = QtGui.QApplication(sys.argv)
        super(ImageViewer, self).__init__()

        #TODO: Add ImageViewer to skimage.io window manager

        self.overlay_cmap = clear_red

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
        self.image_plot = self.ax.images[0]
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

        #TODO: Add coordinate display
        # self.statusBar().showMessage("coordinates")
        self.original_image = image
        self.image = image

        self.overlay_plot = None
        self._overlay = None

    @property
    def image(self):
        return self._img

    @image.setter
    def image(self, image):
        self._img = image
        self.ax.images[0].set_array(image)
        self.canvas.draw_idle()

    @property
    def overlay(self):
        return self._overlay

    @overlay.setter
    def overlay(self, image):
        self._overlay = image
        if self.overlay_plot is None:
            self.overlay_plot = self.ax.imshow(image, cmap=self.overlay_cmap)
        else:
            self.overlay_plot.set_array(image)
        self.canvas.draw_idle()

    def closeEvent(self, ce):
        self.close()

    def show(self):
        super(ImageViewer, self).show()
        sys.exit(qApp.exec_())

    @property
    def climits(self):
        return self.image_plot.get_clim()

    @climits.setter
    def climits(self, limits):
        cmin, cmax = limits
        self.image_plot.set_clim(vmin=cmin, vmax=cmax)

    def connect_event(self, event, callback):
        cid = self.canvas.mpl_connect(event, callback)
        return cid

    def disconnect_event(self, callback_id):
        self.canvas.mpl_disconnect(callback_id)

    def add_artist(self, artist):
        self.ax.add_artist(artist)

    def remove_artist(self, artist):
        """Disconnect all artists created by this widget."""
        # There's probably a smarter way to do this.
        for artist_list in self._axes_artists:
            if artist in artist_list:
                artist_list.remove(artist)

    def redraw(self):
        self.canvas.draw_idle()

