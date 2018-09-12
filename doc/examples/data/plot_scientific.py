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
