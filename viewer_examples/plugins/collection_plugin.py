"""
==================================
``CollectionViewer`` with a plugin
==================================

Demo of a CollectionViewer for viewing collections of images with the
`autolevel` rank filter connected as a plugin.

"""
from skimage import data
from skimage.filters import rank
from skimage.morphology import disk

from skimage.viewer import CollectionViewer
from skimage.viewer.widgets import Slider
from skimage.viewer.plugins.base import Plugin


# Wrap autolevel function to make the disk size a filter argument.
def autolevel(image, disk_size):
    return rank.autolevel(image, disk(disk_size))


img_collection = [data.camera(), data.coins(), data.text()]

plugin = Plugin(image_filter=autolevel)
plugin += Slider('disk_size', 2, 8, value_type='int')
plugin.name = "Autolevel"

viewer = CollectionViewer(img_collection)
viewer += plugin

viewer.show()
