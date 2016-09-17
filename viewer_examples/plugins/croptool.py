from skimage import data
from skimage.viewer import ImageViewer
from skimage.viewer.plugins.crop import Crop


image = data.camera()
viewer = ImageViewer(image)
viewer += Crop()
viewer.show()
