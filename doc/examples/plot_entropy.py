"""
=======
Entropy
=======

In information theory, information entropy is the log-base-2 of the number of
possible outcomes for a message.

For an image, local entropy is related to the complexity contained in a given
neighborhood, typically defined by a structuring element. A large number of
various gray levels has a higher entropy than an homogeneous neighborhood.

The entropy filter can detect subtle variations of local gray level distribution.
In the example, the image is composed of two surfaces with two slightly
different distributions.

Image has a uniform random distribution in the range [-14, +14] in the middle of the
image and a uniform random distribution in the range [-15, 15] at
the image borders, both centered at a gray value of 128.

We apply the local entropy measure using a circular structuring element of
radius 10. As a result, one can detect the central square. The radius is
big enough to efficiently sample the local gray level distribution.

In the second example, the local entropy is used to detect image texture.

"""
import matplotlib.pyplot as plt
import numpy as np

from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk

noise_mask = 28 * np.ones((128, 128), dtype=np.uint8)
noise_mask[32:-32, 32:-32] = 30

noise = (noise_mask * np.random.random(noise_mask.shape) - .5 *
         noise_mask).astype(np.uint8)
img = noise + 128

radius = 10
e = entropy(img, disk(radius))

fig, ax = plt.subplots(1, 3, figsize=(8, 5))
ax1, ax2, ax3 = ax.ravel()

ax1.imshow(noise_mask, cmap=plt.cm.gray)
ax1.set_xlabel('Noise mask')
ax2.imshow(img, cmap=plt.cm.gray)
ax2.set_xlabel('Noised image')
ax3.imshow(e)
ax3.set_xlabel('Local entropy ($r=%d$)' % radius)

# second example: texture detection

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
