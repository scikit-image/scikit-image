from skimage import data
from skimage.viewer import ImageViewer
from skimage.viewer.plugins.canny import CannyPlugin


image = data.camera()
viewer = ImageViewer(image)
p = CannyPlugin(viewer)
p.show()
viewer.show()
