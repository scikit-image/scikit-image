from skimage import data
from skimage.viewer import ImageViewer
from skimage.viewer.plugins.lineprofile import LineProfile


image = data.camera()
viewer = ImageViewer(image)
LineProfile(viewer)
viewer.show()
