from skimage import data
from skimage.filter import canny

from skimage.viewer import ImageViewer
from skimage.viewer.widgets import Slider
from skimage.viewer.plugins.overlayplugin import OverlayPlugin


image = data.camera()
# Note: ImageViewer must be called before Plugin b/c it starts the event loop.
viewer = ImageViewer(image)
# You can create a UI for a filter just by passing a filter function...
plugin = OverlayPlugin(image_filter=canny)
# ... and adding widgets to adjust parameter values.
plugin += Slider('sigma', 0, 5, update_on='release')
plugin += Slider('low threshold', 0, 255, update_on='release')
plugin += Slider('high threshold', 0, 255, update_on='release')
# Finally, attach the plugin to the image viewer.
viewer += plugin
viewer.show()
