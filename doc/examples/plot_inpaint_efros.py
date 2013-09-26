"""
=========================
Textural Image Inpainting
=========================

Image Inpainting is the process of reconstructing lost or deteriorated parts
of an image.

In this example we wll show Textural inpainting. Textures have repetitive
patterns and hence cannot be restored by continuing neighbouring geometric
properties into the unknown region. The correct match is found using the
minimum Sum of Squared Differences (SSD) between a patch about the pixel to
be inpainted and all other patches in the image which do not contain any
boundary region and no unknown or masked region. This implementation
updates 1 pixel at a time.

"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.filter.inpaint_texture import inpaint_efros


image = data.camera()[300:500, 350:550]
mask = np.zeros_like(image, dtype=np.uint8)
paint_region = (slice(125, 145), slice(20, 50))

image[paint_region] = 0
mask[paint_region] = 1

painted = inpaint_efros(image, mask, window=7)

fig, (ax0, ax1) = plt.subplots(ncols=2)
ax0.set_title('Input image')
ax0.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Inpainted image')
ax1.imshow(painted, cmap=plt.cm.gray)
plt.show()
