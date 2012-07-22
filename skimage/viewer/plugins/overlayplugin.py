from ..utils import clear_red

from .base import Plugin


class OverlayPlugin(Plugin):
    """Plugin for ImageViewer that displays an overlay on top of main image.

    Attributes
    ----------
    overlay : array
        Overlay displayed on top of image. This overlay defaults to a color map
        with alpha values varying linearly from 0 to 1.
    """

    def __init__(self, image_viewer, **kwargs):
        Plugin.__init__(self, image_viewer, **kwargs)
        self.overlay_cmap = clear_red
        self._overlay_plot = None
        self._overlay = None

    @property
    def overlay(self):
        return self._overlay

    @overlay.setter
    def overlay(self, image):
        self._overlay = image
        ax = self.image_viewer.ax
        if image is None:
            ax.images.remove(self._overlay_plot)
            self._overlay_plot = None
        elif self._overlay_plot is None:
            self._overlay_plot = ax.imshow(image, cmap=self.overlay_cmap)
        else:
            self._overlay_plot.set_array(image)
        self.image_viewer.redraw()

    def closeEvent(self, event):
        self.overlay = None
        super(OverlayPlugin, self).closeEvent(event)
