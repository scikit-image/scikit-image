from .base import Plugin
from skimage.filter import canny


class CannyPlugin(Plugin):

    def __init__(self, parent, *args, **kwargs):
        height = kwargs.get('height', 100)
        width = kwargs.get('width', 400)
        super(CannyPlugin, self).__init__(self.callback, parent=parent,
                                          width=width, height=height)
        self.add_keyword_argument('sigma', 0.005, 0, self.caller)
        self.add_keyword_argument('low_threshold', 0.255, 0, self.caller)
        self.add_keyword_argument('high_threshold', 0.255, 0, self.caller)
        # Call callback so that image is updated to slider values.
        self.caller()

    def callback(self, *args, **kwargs):
        image = canny(*args, **kwargs)
        self._viewer.overlay = image
