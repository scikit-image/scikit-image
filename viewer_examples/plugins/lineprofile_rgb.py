from skimage import data
from skimage.viewer import ImageViewer
from skimage.viewer.plugins.lineprofile import LineProfile


image = data.chelsea()
viewer = ImageViewer(image)
viewer += LineProfile()
line, rgb_profiles = viewer.show()[0]
