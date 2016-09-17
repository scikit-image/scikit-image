from skimage import data
from skimage.viewer import ImageViewer
from skimage.viewer.plugins.measure import Measure


image = data.camera()
viewer = ImageViewer(image)
viewer += Measure()
viewer.show()
