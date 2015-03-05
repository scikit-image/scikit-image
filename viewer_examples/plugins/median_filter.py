from skimage import data
from skimage.filters.rank import median
from skimage.morphology import disk

from skimage.viewer import ImageViewer
from skimage.viewer.widgets import Slider, OKCancelButtons, SaveButtons
from skimage.viewer.plugins.base import Plugin

def median_filter(image, radius):
    return median(image, selem=disk(radius))

image = data.coins()
viewer = ImageViewer(image)

plugin = Plugin(image_filter=median_filter)
plugin += Slider('radius', 2, 10, value_type='int')
plugin += SaveButtons()
plugin += OKCancelButtons()

viewer += plugin
viewer.show()
