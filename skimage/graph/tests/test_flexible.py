import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
import skimage.graph.mcp as mcp
from skimage import data, img_as_float

from skimage._shared.testing import assert_array_equal

src = img_as_float(data.camera()[128:256, 128:256])
src = src + 0.05 * np.random.standard_normal(src.shape)
target = np.roll(src, (15, -10), axis=(0, 1))
target = target + 0.05 * np.random.standard_normal(target.shape)
src_freq = fft.fftn(src)
target_freq = fft.fftn(target)

# current implementation
image_product1 = src_freq * target_freq.conj()
cross_correlation1 = fft.ifftn(image_product1)

# fixed implementation
image_product = image_product1 / np.abs(image_product1)
cross_correlation = fft.ifftn(image_product)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(np.abs(cross_correlation1), cmap=plt.cm.gray)
axes[0].set_title('Existing Implementation')
axes[1].imshow(np.abs(cross_correlation), cmap=plt.cm.gray)
axes[1].set_title('Proposed Implementation')
for ax in axes:
    ax.set_axis_off()
plt.tight_layout()
plt.show()

 
