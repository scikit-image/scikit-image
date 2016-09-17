"""
==============================================
``CollectionViewer`` with an ``OverlayPlugin``
==============================================

Demo of a CollectionViewer for viewing collections of images with an
overlay plugin.

"""
from skimage import data

from skimage.viewer import CollectionViewer
from skimage.viewer.plugins.canny import CannyPlugin


img_collection = [data.camera(), data.coins(), data.text()]

viewer = CollectionViewer(img_collection)
viewer += CannyPlugin()

viewer.show()
