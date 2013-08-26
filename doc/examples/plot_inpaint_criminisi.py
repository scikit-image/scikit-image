"""
========================================
Structural and Textural Image Inpainting
========================================

Image Inpainting is the process of reconstructing lost or deteriorated parts
of an image.

In this example we wll show Linear structural and Textural inpainting. The
order in which the unknown region is filled is determined by the amount of
reliable information (values already known are denoted as 1 - completely
certain, decreasing as we inpaint further) and amount of data (gradient
strength) surrounding the pixel in question. This implementation updates 1
patch at a time.

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

# For best results, `window` should be larger in size than the texel (texture
# element) being inpainted. For example, in this case, the single white/black
# square is the texel which is of `(25, 25)` shape. A value larger than this
# yields perfect reconstruction, but a value smaller than this, may have couple
# of pixels off.
painted = inpaint_criminisi(image, mask, window=27, max_thresh=0.2)

fig, (ax0, ax1) = plt.subplots(ncols=2)
ax0.set_title('Input image')
ax0.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Inpainted image')
ax1.imshow(painted, cmap=plt.cm.gray)
plt.show()
