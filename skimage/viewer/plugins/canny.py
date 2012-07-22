from skimage.filter import canny

from .overlayplugin import OverlayPlugin
from ..widgets import Slider, ComboBox


class CannyPlugin(OverlayPlugin):

    name = 'Canny Filter'

    def __init__(self, image_viewer, *args, **kwargs):
        height = kwargs.get('height', 100)
        width = kwargs.get('width', 400)
        super(CannyPlugin, self).__init__(image_viewer,
                                          width=width, height=height)
        self.add_widget(Slider('sigma', 0, 5, update_on='release'))
        self.add_widget(Slider('low threshold', 0, 255, update_on='release'))
        self.add_widget(Slider('high threshold', 0, 255, update_on='release'))
        self.add_widget(ComboBox('color', self.color_names, ptype='plugin'))
        # Update image overlay to default slider values.
        self.filter_image()

    def image_filter(self, *args, **kwargs):
        image = canny(*args, **kwargs)
        self.overlay = image
