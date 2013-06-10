"""
ImageViewer class for viewing and interacting with images.
"""
from ..qt import QtGui
from ..qt import QtCore

from skimage import io, img_as_float
from skimage.util.dtype import dtype_range
from skimage.exposure import rescale_intensity
import numpy as np
from .. import utils
from ..widgets import Slider
from ..utils import dialogs


__all__ = ['ImageViewer', 'CollectionViewer']


def mpl_image_to_rgba(mpl_image):
    """Return RGB image from the given matplotlib image object.

    Each image in a matplotlib figure has it's own colormap and normalization
    function. Return RGBA (RGB + alpha channel) image with float dtype.
    """
    input_range = (mpl_image.norm.vmin, mpl_image.norm.vmax)
    image = rescale_intensity(mpl_image.get_array(), in_range=input_range)
    image = mpl_image.cmap(img_as_float(image)) # cmap complains on bool arrays
    return img_as_float(image)


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
        self.file_menu.addAction('Open file', self.open_file,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_O)
        self.file_menu.addAction('Save to file', self.save_to_file,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_S)
        self.file_menu.addAction('Quit', self.close,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.main_widget = QtGui.QWidget()
        self.setCentralWidget(self.main_widget)

        self.fig, self.ax = utils.figimage(image)
        self.canvas = self.fig.canvas
        self.canvas.setParent(self)

        self.ax.autoscale(enable=False)

        self._image_plot = self.ax.images[0]

        self.original_image = image
        self.image = image.copy()
        self.plugins = []

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

    def open_file(self):
        """Open image file and display in viewer."""
        filename = dialogs.open_file_dialog()
        if filename is None:
            return
        image = io.imread(filename)
        self.original_image = image     # update saved image
        self.image = image              # update displayed image

    def save_to_file(self):
        """Save current image to file.

        The current behavior is not ideal: It saves the image displayed on
        screen, so all images will be converted to RGB, and the image size is
        not preserved (resizing the viewer window will alter the size of the
        saved image).
        """
        filename = dialogs.save_file_dialog()
        if filename is None:
            return
        if len(self.ax.images) == 1:
            io.imsave(filename, self.image)
        else:
            underlay = mpl_image_to_rgba(self.ax.images[0])
            overlay = mpl_image_to_rgba(self.ax.images[1])
            alpha = overlay[:, :, 3]

            # alpha can be set by channel of array or by a scalar value.
            # Prefer the alpha channel, but fall back to scalar value.
            if np.all(alpha == 1):
                alpha = np.ones_like(alpha) * self.ax.images[1].get_alpha()

            alpha = alpha[:, :, np.newaxis]
            composite = (overlay[:, :, :3] * alpha +
                         underlay[:, :, :3] * (1 - alpha))
            io.imsave(filename, composite)

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

        # Adjust size if new image shape doesn't match the original
        h, w = image.shape[:2]
        # update data coordinates (otherwise pixel coordinates are off)
        self._image_plot.set_extent((0, w, h, 0))
        # update display (otherwise image doesn't fill the canvas)
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)

        # update color range
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
        for plugin in self.plugins:
            plugin.arguments[0] = self.image
            plugin.filter_image()              # updates self.image

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
        else:
            event.ignore()
