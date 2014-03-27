"""
===========
Convex Hull
===========

The convex hull of a binary image is the set of pixels included in the
smallest convex polygon that surround all white pixels in the input.

In this example, we show how the input pixels (white) get filled in by the
convex hull (white and grey).

A good overview of the algorithm is given on `Steve Eddin's blog
<http://blogs.mathworks.com/steve/2011/10/04/binary-image-convex-hull-algorithm-notes/>`__.

"""
import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import convex_hull_image


image = np.array(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 1, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 1, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)

original_image = np.copy(image)

chull = convex_hull_image(image)
image[chull] += 1
# image is now:
#[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
# [ 0.  0.  0.  0.  2.  0.  0.  0.  0.]
# [ 0.  0.  0.  2.  1.  2.  0.  0.  0.]
# [ 0.  0.  2.  1.  1.  1.  2.  0.  0.]
# [ 0.  2.  1.  1.  1.  1.  1.  2.  0.]
# [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
ax1.set_title('Original picture')
ax1.imshow(original_image, cmap=plt.cm.gray, interpolation='nearest')
ax2.set_title('Transformed picture')
ax2.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
plt.show()
