from skimage.filter import canny
from .overlayplugin import OverlayPlugin


class CannyPlugin(OverlayPlugin):

    name = 'Canny Filter'

    def __init__(self, image_viewer, *args, **kwargs):
        height = kwargs.get('height', 100)
        width = kwargs.get('width', 400)
        super(CannyPlugin, self).__init__(image_viewer,
                                          width=width, height=height)
        self.add_keyword_argument('sigma', 0.005, 0, self.caller,
                                  update_on='release')
        self.add_keyword_argument('low_threshold', 0.255, 0, self.caller,
                                  update_on='release')
        self.add_keyword_argument('high_threshold', 0.255, 0, self.caller,
                                  update_on='release')
        # Call callback so that image is updated to slider values.
        self.caller()

    def callback(self, *args, **kwargs):
        image = canny(*args, **kwargs)
        self.overlay = image

    def closeEvent(self, event):
        self.overlay = None
        super(CannyPlugin, self).closeEvent(event)
