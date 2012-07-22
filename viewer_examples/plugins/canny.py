from skimage import data
from skimage.viewer import ImageViewer
from skimage.viewer.plugins.canny import CannyPlugin


image = data.camera()
viewer = ImageViewer(image)
CannyPlugin(viewer)
viewer.show()
