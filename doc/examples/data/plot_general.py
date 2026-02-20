"""
======================
General-purpose images
======================

The title of each image indicates the name of the function.

"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import skimage as ski

matplotlib.rcParams['font.size'] = 18

images = (
    'astronaut',
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
    caller = getattr(ski.data, name)
    image = caller()
    plt.figure()
    plt.title(name)
    if image.ndim == 2:
        plt.imshow(image, cmap=plt.cm.gray)
    else:
        plt.imshow(image)

plt.show()


############################################################################
# Thumbnail image for the gallery

# sphinx_gallery_thumbnail_number = -1
fig, axs = plt.subplots(nrows=3, ncols=3)
for ax in axs.flat:
    ax.axis("off")
axs[0, 0].imshow(ski.data.astronaut())
axs[0, 1].imshow(ski.data.binary_blobs(), cmap=plt.cm.gray)
axs[0, 2].imshow(ski.data.brick(), cmap=plt.cm.gray)
axs[1, 0].imshow(ski.data.colorwheel())
axs[1, 1].imshow(ski.data.camera(), cmap=plt.cm.gray)
axs[1, 2].imshow(ski.data.cat())
axs[2, 0].imshow(ski.data.checkerboard(), cmap=plt.cm.gray)
axs[2, 1].imshow(ski.data.clock(), cmap=plt.cm.gray)
further_img = np.full((300, 300), 255)
for xpos in [100, 150, 200]:
    further_img[150 - 10 : 150 + 10, xpos - 10 : xpos + 10] = 0
axs[2, 2].imshow(further_img, cmap=plt.cm.gray)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
