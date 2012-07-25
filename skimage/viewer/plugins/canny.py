from skimage.filter import canny

from .overlayplugin import OverlayPlugin
from ..widgets import Slider, ComboBox


class CannyPlugin(OverlayPlugin):
    """Canny filter plugin to show edges of an image."""

    name = 'Canny Filter'

    def __init__(self, *args, **kwargs):
        super(CannyPlugin, self).__init__(image_filter=canny, **kwargs)

        self.add_widget(Slider('sigma', 0, 5, update_on='release'))
        self.add_widget(Slider('low threshold', 0, 255, update_on='release'))
        self.add_widget(Slider('high threshold', 0, 255, update_on='release'))
        self.add_widget(ComboBox('color', self.color_names, ptype='plugin'))
