"""
===========
HDR Image
===========

A HDR (High Dynamic Range) image is a combination of bracketed images (varying
exposures) into one.

In this example, we show the use of a series of images at different exposures
to create a HDR image.

References
----------

.. [1] Debevec and Malik, J. "Recovering high dynamic range radiance maps from
       photographs" (1997). DOI:10.1145/258734.258884

.. [2] https://en.wikipedia.org/wiki/High-dynamic-range_imaging
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.exposure import adjust_gamma, hdr
from skimage import data

# Get example images
ims, exp = data.hdr_images()
exp = np.array(exp)

# Get radiance map (how the radiance maps to the counts for each channel)
radiance_map = hdr.get_crf(ims, exp, depth=8, l=100)

# Show radiance map
plt.title('Camera response function')
plt.xlabel('Counts')
plt.ylabel('Radiance')
for ii, color in enumerate(['Red', 'Green', 'Blue']):
    plt.plot(radiance_map[:, ii], color[0].lower(), label=color)
plt.legend()
plt.show()

# Make the HDR image
hdr_im = hdr.make_hdr(ims, exp, radiance_map, depth=8)

# Normalise the hdr image
hdr_norm = hdr_im / np.nanmax(hdr_im)


fig, axes = plt.subplots(nrows=1, ncols=2)
# Show hdr image. This is going to be dark due to the range in the image
axes[0].imshow(hdr_norm)
axes[0].set_title("HDR image")
axes[0].set_axis_off()
# Show gamma adjusted hdr image.
axes[1].imshow(adjust_gamma(hdr_norm, gamma=0.25))
axes[1].set_title("HDR image gamma adjusted")
axes[1].set_axis_off()
plt.show()

# Below follows a commented out example for saving the image as a hdr image
# importable by other processing software
"""
.. code-block:: python
   from skimage.io import imsave
  imsave(fname, hdr_norm.astype(np.float32), plugin='tifffile')

   # Plotting a histogram equalised  hdr image.
   from matplotlib.colors import LogNorm
   from skimage.exposure import equalize_hi

   tone_mapped = np.zeros_like(hdr_im)
   for ii in range(3):
       tone_mapped[..., ii] = equalize_hist(np.nan_to_num(hdr_im[..., ii]))

   fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
   axes[0].imshow(hdr_im, 'gray', norm=LogNorm())
   axes[0].set_title("HDR on log norm")
   axes[1].imshow(tone_mapped)
   axes[1].set_title("Tone mapped")
   plt.show()
"""
