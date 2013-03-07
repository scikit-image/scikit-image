from skimage import data
from skimage.filter import canny

from skimage.viewer import ImageViewer
from skimage.viewer.widgets import Slider
from skimage.viewer.widgets.history import SaveButtons
from skimage.viewer.plugins.overlayplugin import OverlayPlugin


image = data.camera()

# You can create a UI for a filter just by passing a filter function...
plugin = OverlayPlugin(image_filter=canny)
# ... and adding widgets to adjust parameter values.
plugin += Slider('sigma', 0, 5, update_on='release')
plugin += Slider('low threshold', 0, 255, update_on='release')
plugin += Slider('high threshold', 0, 255, update_on='release')
# ... and we can also add buttons to save the overlay:
plugin += SaveButtons(name='Save overlay to:')

# Finally, attach the plugin to an image viewer.
viewer = ImageViewer(image)
viewer += plugin
viewer.show()
