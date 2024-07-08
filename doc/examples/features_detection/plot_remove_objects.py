"""
================
Removing objects
================

scikit-image has several ways of removing objects from N-dimensional images.
Here, "objects" (and "holes") are defined as groups of samples with the same
label value which distinct from the background and other objects.

This example shows how to remove objects based on their size, or their
distances from other objects.
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
spaced_objects = ski.morphology.remove_objects_by_distance(objects, min_distance=100)

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
