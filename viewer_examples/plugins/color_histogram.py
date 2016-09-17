from skimage.viewer import ImageViewer
from skimage.viewer.plugins.color_histogram import ColorHistogram
from skimage import data


image = data.load('color.png')
viewer = ImageViewer(image)
viewer += ColorHistogram(dock='right')
viewer.show()
