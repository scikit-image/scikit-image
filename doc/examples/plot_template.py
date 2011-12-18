"""
=================
Template Matching
=================

In this example, we use template matching to identify the occurrence of an
object in an image. The ``match_template`` function uses normalised correlation
techniques to find instances of the "target image" in the "test image".

The output of ``match_template`` is an image where we can easily identify peaks
by eye. We mark the locations of matches (red dots), which are detected using
a simple peak extraction algorithm.
"""

import numpy as np
from skimage.feature import match_template
from numpy.random import randn
import matplotlib.pyplot as plt

# We first construct a simple image target:
size = 100
target = np.tri(size) + np.tri(size)[::-1]
# place target in an image at two positions, and add noise.
image = np.zeros((400, 400))
target_positions = [(50, 50), (200, 200)]
for x, y in target_positions:
    image[x:x+size, y:y+size] = target
image += randn(400, 400)*2

# Match the template.
result = match_template(image, target, method='norm-corr')

# peak extraction algorithm.
delta = 5
found_positions = []
for i in range(50):
    index = np.argmax(result)
    y, x = np.unravel_index(index, result.shape)
    if not found_positions:
        found_positions.append((x, y))
    for position in found_positions:
        distance = np.sqrt((x - position[0]) ** 2 + (y - position[1]) ** 2)
        if distance > delta:
            found_positions.append((x, y))
    result[y, x] = 0
    if len(found_positions) == len(target_positions):
        break
x_found, y_found = np.transpose(found_positions)

plt.gray()

plt.subplot(1, 3, 1)
plt.imshow(target)
plt.title("Target image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(image)
plt.title("Test image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(result)
plt.plot(x_found, y_found, 'ro')
plt.title("Result from\n``match_template``")
plt.autoscale(tight=True)
plt.axis('off')

plt.show()
