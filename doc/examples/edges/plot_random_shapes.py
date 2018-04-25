"""
=============
Random Shapes
=============

Example of generating random shapes with particular properties.
"""

import matplotlib.pyplot as plt

from skimage.draw import random_shapes

# Let's start simple and generate a 128x128 image
# with a single grayscale rectangle.
result = random_shapes((128, 128), max_shapes=1, shape='rectangle',
                       num_channels=1)

# We get back a tuple consisting of (1) the image with the generated shapes
# and (2) a list of label tuples with the kind of shape (e.g. circle,
# rectangle) and ((r0, r1), (c0, c1)) coordinates.
image, labels = result
print('Image shape: {}\nLabels: {}'.format(image.shape, labels))

# We can visualize the images.
fig, axes = plt.subplots(nrows=2, ncols=3)
ax = axes.ravel()
ax[0].imshow(image.squeeze(), cmap='gray')
ax[0].set_title('Grayscale shape')

# The generated images can be much more complex. For example, let's try many
# shapes of any color. If we want the colors to be particularly light, we can
# set the min_pixel_intensity to a high value from the range [0,255].
image1, _ = random_shapes((128, 128), max_shapes=10, min_pixel_intensity=100)

# Moar :)
image2, _ = random_shapes((128, 128), max_shapes=10, min_pixel_intensity=200)
image3, _ = random_shapes((128, 128), max_shapes=10, min_pixel_intensity=50)
image4, _ = random_shapes((128, 128), max_shapes=10, min_pixel_intensity=0)

for i, image in enumerate([image1, image2, image3, image4], 1):
    ax[i].imshow(image)
    ax[i].set_title('Colored shapes, #{}'.format(i-1))

# These shapes are well suited to test segmentation algorithms. Often, we
# want shapes to overlap to test the algorithm. This is also possible:
image, _ = random_shapes((128, 128), min_shapes=5, max_shapes=10,
                         min_size=20, allow_overlap=True)
ax[5].imshow(image)
ax[5].set_title('Overlapping shapes')

for a in ax:
    a.set_xticklabels([])
    a.set_yticklabels([])

plt.show()
