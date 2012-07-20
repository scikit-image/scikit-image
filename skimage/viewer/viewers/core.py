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

        self.layout = QtGui.QVBoxLayout(self.main_widget)
        self.layout.addWidget(self.canvas)

        #TODO: Add coordinate display
        # self.statusBar().showMessage("coordinates")
        self.original_image = image
        self.image = image
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
        if len(self.ax.images) == 1:
            self.ax.imshow(image, cmap=self.overlay_cmap)
        else:
            self.ax.images[1].set_array(image)
        self.canvas.draw_idle()

    def closeEvent(self, ce):
        self.close()

    def show(self):
        super(ImageViewer, self).show()
        sys.exit(qApp.exec_())


