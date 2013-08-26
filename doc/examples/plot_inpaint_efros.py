"""This is an example for constrained texture synthesis. An unknown region in
the image is filled using texture of surrounding region. This implementation
updates pixel-by-pixel.

Outline of the algorithm for Texture Synthesis is as follows:
- Loop: Generate the boundary pixels of the region to be inpainted
    - Loop: Generate a template of (window, window), center: boundary pixel
        - Compute the SSD between template and similar sized patches across
          the image
        - Find the pixel with smallest SSD, such that patch isn't where
          template is located (False positive)
        - Update the intensity value of center pixel of template as the
          value of the center of the matched patch
    - Repeat for all pixels of the boundary
- Repeat until all pixels are inpainted

For further information refer to [1]_

References
---------
.. [1] A. Efros and T. Leung. "Texture Synthesis by Non-Parametric
       Sampling". In Proc. Int. Conf. Computer Vision, pages 1033-1038,
       Kerkyra, Greece, September 1999.
       http://graphics.cs.cmu.edu/people/efros/research/EfrosLeung.html

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
