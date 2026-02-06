"""
=====================
Circular Thresholding
=====================

Circular thresholding is a special case of thresholding for circular signals
(e.g. hue values) to create a binary image from a grayscale image [1]_. The
implementation is based on the method proposed by Yu-Kun Lai and Paul L. Rosin [2]_
([preprint PDF](https://users.cs.cf.ac.uk/Yukun.Lai/papers/thresholdingTIP.pdf)).


.. [1] https://en.wikipedia.org/wiki/Circular_thresholding
.. [2] https://ieeexplore.ieee.org/document/6698338
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.color.colorconv import hsv2rgb
from skimage.filters import threshold_circular_otsu, threshold_otsu

#########################################################################
# In this example, the task is to separate the disc in the foreground
# from the background. Both disc and background have a similar hue which
# slightly shifts from one row to the next (left column).
# The two red lines in the histograms of the center column show the two
# threshold values of the circular Otsu method; the dashed blue line
# shows the threshold of the normal Otsu algorithm which has often
# cannot correctly separate the disc from the background.
# The right column shows the binary masks obtained with thresholds of the
# circular Otsu algorithm.

mask = np.fromfunction(lambda r, c: (r - 32) ** 2 + (c - 32) ** 2 < 300, (65, 65))

fig, ax = plt.subplots(5, 3, figsize=(10, 10))
for i in range(5):
    img_hsv = np.ones((*mask.shape, 3), dtype=np.float32)
    hue = img_hsv[..., 0]
    hue[...] = np.where(mask, 0.8, 0.9)
    hue += 0.05 * i
    hue += np.random.normal(0, 0.03, mask.shape)
    hue %= 1.0
    img_rgb = hsv2rgb(img_hsv)

    ax[i, 0].imshow(img_rgb)
    ax[i, 0].axis("off")

    c, x = np.histogram(hue, 256, (0, 1))
    t = threshold_circular_otsu(hue, val_range=(0, 1))
    # equivalent:
    # t = threshold_circular_otsu(val_range=(0, 1), hist=c)
    for v in t:
        ax[i, 1].axvline(v, c="#f00", lw=2)
    ax[i, 1].axvline(threshold_otsu(hue), c="#00f", ls="dashed", lw=2)
    ax[i, 1].plot(0.5 * (x[1:] + x[:-1]), c, color="#000")

    ax[i, 2].imshow((hue < t[0]) | (hue > t[1]), cmap="gray")
    ax[i, 2].axis("off")

plt.tight_layout()
plt.show()
