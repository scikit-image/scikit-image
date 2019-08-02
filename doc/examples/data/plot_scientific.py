"""
=================
Scientific images
=================

The title of each image indicates the name of the function.

"""
import matplotlib.pyplot as plt
import matplotlib

from skimage import data

matplotlib.rcParams['font.size'] = 18

images = ('hubble_deep_field',
          'immunohistochemistry',
          'microaneurysms',
          'moon',
          'retina',
          'shepp_logan_phantom',
          'cell',
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
