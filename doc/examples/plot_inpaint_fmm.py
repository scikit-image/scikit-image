import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.filter import inpaint
plt.gray()


def demo_inpaint():
    image = data.camera()
    paint_region = (slice(240, 260), slice(360, 420))
    paint_region1 = (slice(430, 450), slice(360, 410))

    mask = np.zeros_like(image, dtype=np.uint8)
    mask[paint_region] = 1
    mask[paint_region1] = 1

    return image, inpaint.inpaint_fmm(image, mask)


image, painted = demo_inpaint()

fig, (ax0, ax1) = plt.subplots(ncols=2)
ax0.imshow(image)
ax1.imshow(painted)
plt.show()
