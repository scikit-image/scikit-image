"""
========================================
Structural and Textural Image Inpainting
========================================

Image Inpainting is the process of reconstructing lost or deteriorated parts
of an image.

In this example we wll show linear structural and textural inpainting. There
are two terms which play a role in determining the order of inpainting:

- ``confidence`` signifies the amount of reliable information in the
  neighbourhood. Already known pixels have ``confidence = 1`` and unknown
  pixels ``0``.
- ``data`` term represents the presence of strong edges hitting the mask
  boundary. This implementation updates 1 patch at a time.

"""
import numpy as np
from skimage.filter.inpaint_exemplar import inpaint_criminisi
import matplotlib.pyplot as plt
from skimage.data import checkerboard
from skimage.util import img_as_ubyte


image = img_as_ubyte(checkerboard())
mask = np.zeros_like(image, dtype=np.uint8)
paint_region = (slice(50, 150), slice(50, 150))
image[paint_region] = 0
mask[paint_region] = 1

# For best results, ``window`` should be larger in size than the largest texel
# (texture element) being inpainted. A texel is the smallest repeating block
# of pixels in a texture or pattern. For example, in the case below of the
# ``skimage.data.checkerboard`` image, the single white/black square is the
# largest texel which is of shape ``(25, 25)``. A value larger than this yields
# perfect reconstruction, but in case of a value smaller than this perfect
# reconstruction may not be possible.

painted = inpaint_criminisi(image, mask, window=27, ssd_thresh=0.2)

fig, (ax0, ax1) = plt.subplots(ncols=2)
ax0.set_title('Input image')
ax0.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Inpainted image')
ax1.imshow(painted, cmap=plt.cm.gray)
plt.show()
