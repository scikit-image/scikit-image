"""
=======
Entropy
=======

Image entropy is a quantity which is used to describe the amount of information
coded in an image.

"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.filter.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte


image = img_as_ubyte(data.camera())

plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.imshow(image, cmap=plt.cm.gray)
plt.title('Image')
plt.colorbar()

plt.subplot(122)
plt.imshow(entropy(image, disk(5)), cmap=plt.cm.jet)
plt.title('Entropy')
plt.colorbar()

plt.show()
