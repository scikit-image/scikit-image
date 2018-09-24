"""
===================================
Masked Normalized Cross-Correlation
===================================

In this example, we use the masked normalized cross-correlation to identify the relative shift
between two similar images containing invalid data.

In this case, the images cannot simply be masked before computing the cross-correlation, 
as the masks will influence the computation. The influence of the masks must be removed from
the cross-correlation, as is described in [1]_.

In this example, we register the translation between two images. However, one of the 
images has about 25% of the pixels which are corrupted.

.. [1] D. Padfield, "Masked object registration in the Fourier domain" 
       IEEE Transactions on Image Processing (2012). :DOI:`10.1109/TIP.2011.2181402`

"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.feature import masked_register_translation
from scipy import ndimage

image = data.camera()
shift = (-22, 13)

#############################################
# Define areas of the image which are invalid.
# Probability of an invalid pixel is 25%.
# This could be due to a faulty detector, or edges that
# are not affected by translation (e.g. moving object in a window).
# See reference paper for more examples
corrupted_pixels = np.random.choice([False, True], 
                                    size = image.shape, 
                                    p = [0.25, 0.75])

# The shift corresponds to the pixel offset relative to the reference image
offset_image = ndimage.shift(image, shift)
offset_image *= corrupted_pixels
print("Known offset (y, x): {}".format(shift))

# Determine what the mask is based on which pixels are invalid
# In this case, we know what the mask should be since we corrupted 
# the pixels ourselves
mask = corrupted_pixels

shift = masked_register_translation(image, offset_image, mask)

print("Detected pixel offset (y, x): {}".format(shift))

fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)

ax1.imshow(image, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Reference image')

ax2.imshow(offset_image.real, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Corrupted, offset image')

ax3.imshow(mask, cmap='gray')
ax3.set_axis_off()
ax3.set_title('Masked pixels')


plt.show()
