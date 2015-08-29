"""
=======
Entropy
=======

The following example applies the local entropy measure to a noised image with a variable noise.

"""
import matplotlib.pyplot as plt
import numpy as np

from skimage.filters.rank import entropy
from skimage.morphology import disk

noise_mask = 28*np.ones((128, 128), dtype=np.uint8)
noise_mask[32:-32, 32:-32] = 30

noise = (noise_mask*np.random.random(noise_mask.shape)-.5*noise_mask).astype(np.uint8)
img = noise + 128

radius = 10
e = entropy(img, disk(radius))

plt.figure(figsize=[15, 5])
plt.subplot(1, 3, 1)
plt.imshow(noise_mask, cmap=plt.cm.gray)
plt.xlabel('noise mask')
plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(img, cmap=plt.cm.gray)
plt.xlabel('noised image')
plt.colorbar()
plt.subplot(1, 3, 3)
plt.imshow(e)
plt.xlabel('image local entropy ($r=%d$)' % radius)
plt.colorbar()

plt.show()
