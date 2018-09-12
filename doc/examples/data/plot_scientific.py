"""
=================
Scientific images
=================

The title of each image indicates the name of the function.

"""
import matplotlib.pyplot as plt

from skimage import data


images = ('hubble_deep_field',
          'immunohistochemistry',
          'moon',
          )

fig, axes = plt.subplots(len(images), 1, figsize=(8, 4 * len(images)))
ax = axes.ravel()

for i, name in enumerate(images):
    caller = getattr(data, name)
    image = caller()
    ax[i].set_title(name)
    if image.ndim == 2:
        ax[i].imshow(image, cmap=plt.cm.gray)
    else:
        ax[i].imshow(image)

fig.tight_layout()
plt.show()
