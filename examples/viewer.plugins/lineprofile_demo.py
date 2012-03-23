from skimage import data
from skimage.viewer import ImageViewer
from skimage.viewer.plugins import LineProfile

image = data.camera()
view = ImageViewer(image)
LineProfile(view, limits='dtype')
view.show()

