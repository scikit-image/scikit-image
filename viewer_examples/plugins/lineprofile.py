from skimage import data
from skimage.viewer import ImageViewer
from skimage.viewer.plugins.lineprofile import LineProfile


image = data.camera()
viewer = ImageViewer(image)
p = LineProfile(viewer)
p.show()
viewer.show()
