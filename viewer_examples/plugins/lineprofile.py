from skimage import data
from skimage.viewer import ImageViewer
from skimage.viewer.plugins.lineprofile import LineProfile


image = data.camera()
viewer = ImageViewer(image)
viewer += LineProfile()
line, profile = viewer.show()[0]
