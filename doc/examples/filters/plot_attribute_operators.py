"""
===================
Attribute operators
===================
Attribute operators (or connected operators) is a family of contour preserving
filtering operations in mathematical morphology. They can be implemented by
max-trees, a compact hierarchical representation of the image.

Here, we show how to use area and diameter openings.

References
----------
.. [1] Salembier, P., Oliveras, A., & Garrido, L. (1998). Antiextensive
       Connected Operators for Image and Sequence Processing.
       IEEE Transactions on Image Processing, 7(4), 555-570.
.. [2] Carlinet, E., & Geraud, T. (2014). A Comparative Review of
       Component Tree Computation Algorithms. IEEE Transactions on Image
       Processing, 23(9), 3885-3895.
.. [3] Vincent L., Proc. "Grayscale area openings and closings,
       their efficient implementation and applications",
       EURASIP Workshop on Mathematical Morphology and its
       Applications to Signal Processing, Barcelona, Spain, pp.22-27,
       May 1993.
.. [4] Walter, T., & Klein, J.-C. (2002). Automatic Detection of
       Microaneurysms in Color Fundus Images of the Human Retina by Means
       of the Bounding Box Closing. In A. Colosimo, P. Sirabella,
       A. Giuliani (Eds.), Medical Data Analysis. Lecture Notes in Computer
       Science, vol 2526, pp. 210-220. Springer Berlin Heidelberg.

"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import diameter_closing
from skimage import data
from skimage.morphology import closing
from skimage.morphology import square

# image with printed letters
image = data.page()

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Original image
ax[0].imshow(image, cmap='gray', aspect='equal')
ax[0].set_title('Original')
ax[0].axis('off')

# Diameter closing : we remove all dark structures with a maximal
# extension of less than 23. I.e. in closed_attr, all local minima
# have at least a maximal extension of 23.
closed_attr = diameter_closing(image, 23)

# We then calculate the difference to the original image.
tophat_attr = closed_attr - image

ax[1].imshow(tophat_attr, cmap='gray', aspect='equal')
ax[1].set_title('Diameter Closing')
ax[1].axis('off')

# A morphological closing is removing all dark structures that cannot
# contain a structuring element of a certain size.
closed = closing(image, square(12))

# Again we calculate the difference to the original image.
tophat = closed - image

ax[2].imshow(tophat, cmap='gray', aspect='equal')
ax[2].set_title('Morphological Closing')
ax[2].axis('off')

plt.tight_layout()
plt.show()

# Comparing the two results, we observe that the difference between
# image and morphological closing also extracts the long line. A thin
# but long line cannot contain the structuring element. The diameter
# closing stops the filling as soon as a maximal extension is reached.
# The line is therefore not filled and therefore not extracted by the
# difference.
