"""
Basic viewers for viewing images and image collections.
"""
import matplotlib.pyplot as plt

from ..utils import figimage
from ..widgets.slider import Slider


__all__ = ['ImageViewer', 'CollectionViewer']


class ImageViewer(object):
    """Window for displaying images.

    This window is a simple container object that holds a Matplotlib axes
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
    climits : tuple
        Intensity range (minimum, maximum) of *displayed* image. Intensity
        values above and below limits are clipped, but remain in image array.
    plugins : list
        List of attached plugins.
    """

    def __init__(self, image, **kwargs):
        self._image = image.copy()
        self.fig, self.ax = figimage(image, **kwargs)
        self.ax.autoscale(enable=False)
        self.plugins = []

        self.canvas = self.fig.canvas
        if len(self.ax.images) > 0:
            self._imgplot = self.ax.images[0]
            self._img = self._imgplot.get_array()
        else:
            raise ValueError("No image found in figure")

        self.ax.format_coord = self._format_coord

        self._axes_artists = [self.ax.artists,
                              self.ax.collections,
                              self.ax.images,
                              self.ax.lines,
                              self.ax.patches,
                              self.ax.texts]

        self.connect_event('close_event', self.on_close)

    def on_close(self, event):
        # Close all connected plugins
        for p in self.plugins:
            plt.close(p.figure)

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        self._image = image
        self.ax.images[0].set_array(image)

    @property
    def climits(self):
        return self._imgplot.get_clim()

    @climits.setter
    def climits(self, limits):
        cmin, cmax = limits
        self._imgplot.set_clim(vmin=cmin, vmax=cmax)

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

    def _format_coord(self, x, y):
        # callback function to format coordinate display in toolbar
        x = int(x + 0.5)
        y = int(y + 0.5)
        try:
            return "%s @ [%4i, %4i]" % (self.image[y, x], x, y)
        except IndexError:
            return ""

    def show(self):
        plt.show()


class CollectionViewer(ImageViewer):
    """Window for displaying image collections.

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
    update : {'on_slide' | 'on_release'}
        Control whether image is updated on slide or release of the image
        slider. Using 'on_release' will give smoother behavior when displaying
        large images or when writing a plugin/subclass that requires heavy
        computation.
    """

    def __init__(self, image_collection, update='on_slide', **kwargs):
        self.image_collection = image_collection
        self.index = 0
        self.num_images = len(self.image_collection)

        first_image = image_collection[0]
        ImageViewer.__init__(self, first_image, **kwargs)

        h_old = self.fig.get_figheight()
        h_new = h_old + 0.5
        self.fig.set_figheight(h_new)
        self.ax.set_position([0, 1 - h_old/h_new, 1, h_old/h_new])
        ax_slider = self.fig.add_axes([0.1, 0, 0.8, 0.5 / h_new])
        idx_range = (0, self.num_images-1)

        slider_kws = dict(value=0, value_fmt='%i')
        slider_kws[update] = self.update_index
        self.slider = Slider(ax_slider, idx_range, **slider_kws)

        self.connect_event('key_press_event', self.on_keypressed)

    def update_index(self, index):
        """Select image on display using index into image collection."""
        index = int(round(index))

        if index == self.index:
            return

        # clip index value to collection limits
        index = max(index, 0)
        index = min(index, self.num_images-1)

        self.index = index
        self.slider.value = index
        self.update_image(self.image_collection[index])

    def update_image(self, image):
        """Update displayed image.

        This method can be overridden or extended in subclasses and plugins to
        react to image changes.
        """
        self.image = image
        # The following call to draw may be unnecessary.
        self.fig.canvas.draw()

    def on_keypressed(self, event):
        key = event.key
        if str(key) in '0123456789':
            index = 0.1 * int(key) * self.num_images
            self.update_index(index)
        elif key == 'right':
            self.update_index(self.index + 1)
        elif key == 'left':
            self.update_index(self.index - 1)
        elif key == 'end':
            self.update_index(self.num_images - 1)
        elif key == 'home':
            self.update_index(0)

