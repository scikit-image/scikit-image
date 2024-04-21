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
import skimage as ski

# Extract foreground by thresholding an image taken by the Hubble Telescope
image = ski.color.rgb2gray(ski.data.hubble_deep_field())
foreground = image > ski.filters.threshold_li(image)
objects = ski.measure.label(foreground)

# Separate objects into regions larger and smaller than 100 pixels
large_objects = ski.morphology.remove_small_objects(objects, min_size=100)
small_objects = objects ^ large_objects

# Remove objects until remaining ones are at least 100 pixels apart.
# By default, larger ones take precedence.
spaced_objects = ski.morphology.remove_near_objects(objects, min_distance=100)

# Plot the results
fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
ax[0, 0].set_title("labeled objects")
ax[0, 0].imshow(ski.color.label2rgb(objects, image=image, bg_label=0))
ax[0, 1].set_title("large objects")
ax[0, 1].imshow(ski.color.label2rgb(large_objects, bg_label=0))
ax[1, 1].set_title("small objects")
ax[1, 1].imshow(ski.color.label2rgb(small_objects, bg_label=0))
ax[1, 0].set_title("spaced objects (nearest removed)")
ax[1, 0].imshow(ski.color.label2rgb(spaced_objects, bg_label=0))
plt.show()
