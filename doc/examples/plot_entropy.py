"""
=======
Entropy
=======

In information theory, information entropy is the log-base-2 of the number of possible outcomes
for a message.

For an image, local entropy is related to the complexity contained in a given neighborhood, typically defined by a
structuring element. A large number of various gray levels has a higher entropy than an homogeneous neighborhood.

Entropy filter can detect subtle variations of local gray level distribution, in the example, the
image is composed of two surfaces with two slightly different distributions.

Image center has a random distribution in the range [-14,+14] centered on 128, while the borders has a
random distribution in the range [-15,+15] centered on 128.

We apply the local entropy measure using a circular structuring element of radius 10. As a result, one can
detect the central square. Radius should be big enough to efficiently sample the local gray level distribution.

In the second example, the local entropy is used to detect image texture.

"""
import matplotlib.pyplot as plt
import numpy as np

from skimage import data
from skimage.util import img_as_ubyte
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

#second example: texture detection

image = img_as_ubyte(data.camera())

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 4))

img0 = ax0.imshow(image, cmap=plt.cm.gray)
ax0.set_title('Image')
ax0.axis('off')
fig.colorbar(img0, ax=ax0)

img1 = ax1.imshow(entropy(image, disk(5)), cmap=plt.cm.jet)
ax1.set_title('Entropy')
ax1.axis('off')
fig.colorbar(img1, ax=ax1)

plt.show()
