"""
==========
Flood Fill
==========

Flood fill is an algorithm to identify and/or change adjacent values in an
image based on their similarity to an initial seed point [1]_. The conceptual
analogy is the 'paint bucket' tool in many graphic editors.

.. [1] https://en.wikipedia.org/wiki/Flood_fill

Basic example
-------------

First, a basic example where we will change a checkerboard square from white
to mid-gray.
"""

import matplotlib.pyplot as plt
from skimage import data
from skimage.segmentation import flood_fill


checkers = data.checkerboard()

# Fill a square near the middle with value 127, starting at index (76, 76)
filled_checkers = flood_fill(checkers, (76, 76), 127)

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

ax[0].imshow(checkers, cmap=plt.cm.gray, interpolation='none')
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(filled_checkers, cmap=plt.cm.gray, interpolation='none')
ax[1].set_title('After flood fill')
ax[1].axis('off')

plt.show()

"""
.. image:: PLOT2RST.current_figure

Advanced example
----------------

Because standard flood filling requires the neighbors to be strictly equal,
its use is limited on real-world images with color gradients and noise.
The `tolerance` keyword argument widens the permitted range about the initial
value, allowing use on real-world images.

Here we will experiment a bit on the cameraman.  First, turning his coat from
dark to light.
"""

cameraman = data.camera()

# Change the cameraman's coat from dark to light (255).  The seed point is
# chosen as (200, 100),
light_coat = flood_fill(cameraman, (200, 100), 255, tolerance=10)
fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

ax[0].imshow(cameraman, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(light_coat, cmap=plt.cm.gray)
ax[1].set_title('After flood fill')
ax[1].axis('off')

plt.show()


"""
.. image:: PLOT2RST.current_figure

Because the cameraman is dark haired it also changed his hair, as well as
parts of the tripod.

Experimentation with tolerance
------------------------------

To get a better intuitive understanding of how the tolerance parameter works,
here is a set of images progressively increasing the parameter with seed
point in the upper left corner.
"""

output = []

for i in range(8):
    tol = 5 + 20*i
    output.append(flood_fill(cameraman, (0, 0), 255, tolerance=tol))

# Initialize plot and place original image
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
ax[0, 0].imshow(cameraman, cmap=plt.cm.gray)
ax[0, 0].set_title('Original')
ax[0, 0].axis('off')

# Plot all eight different tolerances for comparison.
for i in range(8):
    n = (1 + i)%3
    m = (1 + i)//3
    ax[m, n].imshow(output[i], cmap=plt.cm.gray)
    ax[m, n].set_title('Tolerance {0}'.format(str(5 + 20*i)))
    ax[m, n].axis('off')

fig.tight_layout()
plt.show()

"""
.. image:: PLOT2RST.current_figure

Flood as mask
-------------

A sister function, `flood`, is available which returns a mask identifying the
flood rather than modifying the image itself.  This is useful for
segmentation purposes and more advanced analysis pipelines.
"""
