"""
=================
Template Matching
=================

In this example, we use template matching to identify the occurrence of an
image patch (in this case, a sub-image centered on the camera man's head).
Since there's only a single match, the maximum value in the `match_template`
result` corresponds to the head location. If you expect multiple matches, you
should use a proper peak-finding function.

"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.feature import match_template

image = data.camera()
head = image[70:170, 180:280]

result = match_template(image, head)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))

ax1.imshow(head)
ax1.set_axis_off()
ax1.set_title('template')

ax2.imshow(image)
ax2.set_axis_off()
ax2.set_title('image')

# highlight matched region
xy = np.unravel_index(np.argmax(result), result.shape)[::-1] #-1 flips ij to xy
wface, hface = head.shape
rect = plt.Rectangle(xy, wface, hface, edgecolor='r', facecolor='none')
ax2.add_patch(rect)

plt.show()

