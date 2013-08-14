import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.filter.inpaint_texture import inpaint_texture


image = data.camera()[300:500, 350:550]
mask = np.zeros_like(image, dtype=np.uint8)
paint_region = (slice(125, 145), slice(20, 50))

image[paint_region] = 0
mask[paint_region] = 1

painted = inpaint_texture(image, mask, 7)

fig, (ax0, ax1) = plt.subplots(ncols=2)
ax0.imshow(image, cmap=plt.cm.gray)
ax1.imshow(painted, cmap=plt.cm.gray)
plt.show()
