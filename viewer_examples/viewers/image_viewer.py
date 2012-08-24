from skimage import data
from skimage.viewer import ImageViewer


image = data.camera()
viewer = ImageViewer(image)
viewer.show()
