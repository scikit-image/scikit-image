"""
=====================
CollectionViewer demo
=====================

Demo of CollectionViewer for viewing collections of images. This demo uses
successively darker versions of the same image to fake an image collection.

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
import numpy as np
from skimage import data
from skimage.viewer import CollectionViewer

img = data.lena()
img_collection = [np.uint8(img * 0.9**i) for i in range(20)]

view = CollectionViewer(img_collection)
view.show()
