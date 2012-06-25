from skimage import data
from skimage.viewer import ImageViewer
from skimage.viewer.plugins import ContrastSetter

image = data.coins()
view = ImageViewer(image)
ContrastSetter(view)
view.show()
