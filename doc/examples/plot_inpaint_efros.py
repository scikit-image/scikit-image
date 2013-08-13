import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.filter.inpaint_texture import inpaint_texture


image = data.camera()[400:500, 250:350]
mask = np.zeros_like(image, dtype=np.uint8)
paint_region = (slice(20, 45), slice(40, 60))

image[paint_region] = 0
mask[paint_region] = 1

painted = inpaint_texture(image, mask, 5)

fig, (ax0, ax1) = plt.subplots(ncols=2)
ax0.imshow(image, cmap=plt.cm.gray)
ax1.imshow(painted, cmap=plt.cm.gray)
plt.show()
