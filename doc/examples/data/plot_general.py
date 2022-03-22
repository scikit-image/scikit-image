"""
======================
General-purpose images
======================

The title of each image indicates the name of the function.

"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from skimage import data

matplotlib.rcParams['font.size'] = 18

images = ('astronaut',
          'binary_blobs',
          'brick',
          'colorwheel',
          'camera',
          'cat',
          'checkerboard',
          'clock',
          'coffee',
          'coins',
          'eagle',
          'grass',
          'gravel',
          'horse',
          'logo',
          'page',
          'text',
          'rocket',
          )


for name in images:
    caller = getattr(data, name)
    image = caller()
    plt.figure()
    plt.title(name)
    if image.ndim == 2:
        plt.imshow(image, cmap=plt.cm.gray)
    else:
        plt.imshow(image)

plt.show()


############################################################################
# A thumbnail image for the gallery

fig, axs = plt.subplots(nrows=3, ncols=2)
# The next line sets the thumbnail for the last figure in the gallery
# sphinx_gallery_thumbnail_number = -1
for ax in axs.flat:
    ax.axis("off")
axs[0, 0].imshow(data.astronaut())
axs[0, 1].imshow(data.binary_blobs(), cmap=plt.cm.gray)
axs[1, 0].imshow(data.brick(), cmap=plt.cm.gray)
axs[1, 1].imshow(data.colorwheel())
axs[2, 0].imshow(data.camera(), cmap=plt.cm.gray)
further_img = np.full((300, 300), 255)
for xpos in [100, 150, 200]:
    further_img[150 - 10 : 150 + 10, xpos - 10 : xpos + 10] = 0
axs[2, 1].imshow(further_img, cmap=plt.cm.gray)
plt.subplots_adjust(wspace=-0.7, hspace=0.1)
