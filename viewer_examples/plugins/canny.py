from skimage import data
from skimage.viewer import ImageViewer
from skimage.viewer.plugins.canny import CannyPlugin


image = data.camera()
viewer = ImageViewer(image)
viewer += CannyPlugin()
canny_edges = viewer.show()[0][0]
