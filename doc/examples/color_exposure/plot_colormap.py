"""
================================
Drawing an image with a colormap
================================

This example demonstates the use of the :py:func:`skimage.color.img_to_cmap`
function, which takes an input grayscale image and returns an RGB image drawn
by using the specified colormap.
"""
from skimage import color, data
from matplotlib import pyplot as plt
from matplotlib import colors
from skimage.util.colormap import viridis

img = data.camera()

titles = ['Image drawn with viridis colormap',
          'Image drawn with a dark red to yellow colormap']

my_cmap = colors.LinearSegmentedColormap.from_list('my-map',
                                                   ['darkred', 'yellow'])
cmaps = [viridis, my_cmap]

for i in range(2):
    plt.figure()
    plt.title(titles[i])
    out = color.colormap_image(img, cmaps[i])
    plt.imshow(out)

plt.show()
