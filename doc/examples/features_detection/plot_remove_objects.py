"""
================
Removing objects
================

scikit-image supports several ways to remove objects inside N-dimensional
images. In this context "objects" (and "holes") are defined as groups of
connected samples that are distinct from the background. A binary image can
contain several objects that are not connected to each other.

The code snippet below demonstrates two ways to remove objects inside an image
either by removing objects

- based on the number of samples that make up each object
- or based on their distance of each object to each other.
"""

import matplotlib.pyplot as plt
from skimage import data, morphology, color, filters

# Extract foreground by thresholding an image taken by the Hubble Telescope
image = color.rgb2gray(data.hubble_deep_field())
foreground = image > filters.threshold_li(image)

# Separate objects into regions larger and smaller than 100 pixels
large_objects = morphology.remove_small_objects(foreground, min_size=100)
small_objects = foreground ^ large_objects

# Remove objects until remaining ones are at least 100 pixels apart,
# smaller ones are removed in favor of larger ones by default
spaced_objects = morphology.remove_near_objects(
    foreground, minimal_distance=100
)

# Plot the results
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].set_title("original")
ax[0, 0].imshow(foreground)
ax[0, 1].set_title("large objects")
ax[0, 1].imshow(large_objects)
ax[1, 1].set_title("small objects")
ax[1, 1].imshow(small_objects)
ax[1, 0].set_title("spaced objects (nearest removed)")
ax[1, 0].imshow(spaced_objects)
plt.show()
