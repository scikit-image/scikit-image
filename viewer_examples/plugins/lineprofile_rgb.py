from skimage import data
from skimage.viewer import ImageViewer
from skimage.viewer.plugins.lineprofile import LineProfile


image = data.chelsea()
viewer = ImageViewer(image)
viewer += LineProfile()
line, profiles = viewer.show()[0]
