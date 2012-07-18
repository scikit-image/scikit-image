import skimage
from skimage import data
from skimage.viewer import ImageViewer
from skimage.viewer.plugins.houghexplorer import HoughExplorer

image = skimage.img_as_float(data.camera())
view = ImageViewer(image)
HoughExplorer(view)
view.show()

