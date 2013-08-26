"""
===========================
Structural Image Inpainting
===========================

Image Inpainting is the process of reconstructing lost or deteriorated parts
of an image.

In this example we wll show Structural inpainting which uses geometric
approaches for filling in the missing information in the region which should
be inpainted. These algorithms focus on the consistency of the geometric
structure.

"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.filter import inpaint


image = data.camera()
paint_region = (slice(240, 260), slice(360, 420))
paint_region1 = (slice(430, 450), slice(360, 410))

mask = np.zeros_like(image, dtype=np.uint8)
mask[paint_region] = 1
mask[paint_region1] = 1
image[mask == 1] = 0

painted = inpaint.inpaint_fmm(image, mask)

fig, (ax0, ax1) = plt.subplots(ncols=2)
ax0.set_title("Input Image")
ax0.imshow(image, cmap=plt.cm.gray)
ax1.set_title("Inpainted Image")
ax1.imshow(painted, cmap=plt.cm.gray)
plt.show()
