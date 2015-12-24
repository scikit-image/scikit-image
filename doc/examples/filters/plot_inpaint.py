"""
===========
Inpainting
===========
Inpainting [1]_ is the process of reconstructing lost or deteriorated
parts of images and videos. 

The reconstruction is supposed to be performed in fully automatic way by
exploiting the information presented in non-damaged regions.

In this example, we show how the masked pixels get inpainted by
inpainting algorithm based on 'biharmonic equation'-assumption [2]_ [3]_.

.. [1]  Wikipedia. Inpainting
        https://en.wikipedia.org/wiki/Inpainting
.. [2]  Wikipedia. Biharmonic equation
        https://en.wikipedia.org/wiki/Biharmonic_equation
.. [3]  N.S.Hoang, S.B.Damelin, "On surface completion and image 
        inpainting by biharmonic functions: numerical aspects",
        http://www.ima.umn.edu/~damelin/biharmonic
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.restoration import inpaint

image_orig = color.rgb2gray(data.astronaut())

# Create mask with three defect regions: left, middle, right respectively
mask = np.zeros_like(image_orig)
mask[20:60, 0:20] = 1
mask[200:300, 150:170] = 1
mask[50:100, 400:430] = 1

image_defect = image_orig.copy()
image_defect[np.where(mask)] = 0

image_result = inpaint.inpaint_biharmonic(image_defect, mask)

fig, axes = plt.subplots(ncols=3, nrows=1)

axes[0].set_title('Defected image')
axes[0].imshow(image_orig, cmap=plt.cm.gray, interpolation='nearest')
axes[0].set_xticks([]), axes[0].set_yticks([])

axes[1].set_title('Defect mask')
axes[1].imshow(mask, cmap=plt.cm.gray, interpolation='nearest')
axes[1].set_xticks([]), axes[1].set_yticks([])

axes[2].set_title('Inpainted image')
axes[2].imshow(image_result, cmap=plt.cm.gray, interpolation='nearest')
axes[2].set_xticks([]), axes[2].set_yticks([])

plt.show()
