from skimage import data
from skimage.filter import median_filter

from skimage.viewer import ImageViewer
from skimage.viewer.widgets import Slider
from skimage.viewer.widgets.history import OKCancelButtons, SaveButtons
from skimage.viewer.plugins.base import Plugin


image = data.coins()
viewer = ImageViewer(image)

plugin = Plugin(image_filter=median_filter)
plugin += Slider('radius', 2, 10, value_type='int', update_on='release')
plugin += SaveButtons()
plugin += OKCancelButtons()

viewer += plugin
viewer.show()
