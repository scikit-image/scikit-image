import skimage
from skimage import data
from skimage.viewer import ImageViewer
from skimage.viewer.plugins import EdgeDetector

image = skimage.img_as_float(data.camera())
view = ImageViewer(image)
EdgeDetector(view)
view.show()

