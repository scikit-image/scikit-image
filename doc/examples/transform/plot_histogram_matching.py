"""
==================
Histogram matching
==================

This example demonstrates the feature of histogram matching. It manipulates the
pixels of an input image so that its histogram matches the histogram of the
reference image. If the images have multiple channels, the matching is done
independently for each channel, as long as the number of channels is equal in
the input image and the reference.

Histogram matching can be used as a lightweight normalisation for image
processing, such as feature matching, especially in circumstances where the
images have been taken from different sources or in different conditions (i.e.
lighting).
"""

import matplotlib.pyplot as plt

from skimage import data
from skimage.transform import match_histograms

reference = data.astronaut()
image = data.chelsea()

matched = match_histograms(image, reference)

fig = plt.figure()
gs = plt.GridSpec(2, 3)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)

for aa in (ax1, ax2, ax3):
    aa.set_axis_off()

ax1.imshow(image)
ax1.set_title('Source')
ax2.imshow(reference)
ax2.set_title('Reference')
ax3.imshow(matched)
ax3.set_title('Matched')

plt.tight_layout()
plt.show()
