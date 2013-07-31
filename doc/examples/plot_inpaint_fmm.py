import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.filter import inpaint
from skimage.filter import _inpaint


def demo_inpaint():
    image = data.camera()
    paint_region = (slice(120, 130), slice(440, 470))
    paint_region1 = (slice(220, 230), slice(440, 470))
    image[paint_region] = 0
    image[paint_region1] = 0

    mask = np.zeros_like(image, dtype=np.uint8)
    mask[paint_region] = 1
    mask[paint_region1] = 1

    return image, inpaint.inpaint_fmm(image, mask)


image, painted = demo_inpaint()

fig, (ax0, ax1) = plt.subplots(ncols=2)
ax0.imshow(image)
ax1.imshow(painted)
plt.show()
