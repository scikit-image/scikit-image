from skimage.filter import canny

from .overlayplugin import OverlayPlugin


class CannyPlugin(OverlayPlugin):

    name = 'Canny Filter'

    def __init__(self, image_viewer, *args, **kwargs):
        height = kwargs.get('height', 100)
        width = kwargs.get('width', 400)
        super(CannyPlugin, self).__init__(image_viewer,
                                          width=width, height=height)
        self.add_widget('sigma', 0, 5, update_on='release')
        self.add_widget('low_threshold', 0, 255, update_on='release')
        self.add_widget('high_threshold', 0, 255, update_on='release')
        # Update image overlay to default slider values.
        self.filter_image()

    def image_filter(self, *args, **kwargs):
        image = canny(*args, **kwargs)
        self.overlay = image
