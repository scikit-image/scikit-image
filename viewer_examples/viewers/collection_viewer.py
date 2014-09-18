"""
=====================
CollectionViewer demo
=====================

Demo of CollectionViewer for viewing collections of images. This demo uses
the different layers of the gaussian pyramid as image collection.

You can scroll through images with the slider, or you can interact with the
viewer using your keyboard:

left/right arrows
    Previous/next image in collection.
number keys, 0--9
    0% to 90% of collection. For example, "5" goes to the image in the
    middle (i.e. 50%) of the collection.
home/end keys
    First/last image in collection.

"""
from skimage import data
from skimage.viewer import CollectionViewer
from skimage.transform import pyramid_gaussian


img = data.astronaut()
img_collection = tuple(pyramid_gaussian(img))

view = CollectionViewer(img_collection)
view.show()
