"""
============
Mean filters
============

This example compares the following mean filters of the rank filter package:

 * **local mean**: all pixels belonging to the structuring element to compute
   average gray level.
 * **percentile mean**: only use values between percentiles p0 and p1
   (here 10% and 90%).
 * **bilateral mean**: only use pixels of the structuring element having a gray
   level situated inside g-s0 and g+s1 (here g-500 and g+500)

Percentile and usual mean give here similar results, these filters smooth the
complete image (background and details). Bilateral mean exhibits a high
filtering rate for continuous area (i.e. background) while higher image
frequencies remain untouched.

"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.morphology import disk
from skimage.filters import rank


image = (data.coins()).astype(np.uint16) * 16
selem = disk(20)

percentile_result = rank.mean_percentile(image, selem=selem, p0=.1, p1=.9)
bilateral_result = rank.mean_bilateral(image, selem=selem, s0=500, s1=500)
normal_result = rank.mean(image, selem=selem)


fig, axes = plt.subplots(nrows=3, figsize=(8, 10))
ax0, ax1, ax2 = axes

ax0.imshow(np.hstack((image, percentile_result)))
ax0.set_title('Percentile mean')
ax0.axis('off')

ax1.imshow(np.hstack((image, bilateral_result)))
ax1.set_title('Bilateral mean')
ax1.axis('off')

ax2.imshow(np.hstack((image, normal_result)))
ax2.set_title('Local mean')
ax2.axis('off')

plt.show()
