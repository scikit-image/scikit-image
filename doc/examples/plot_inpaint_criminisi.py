"""This is an example for constrained texture synthesis using Criminisi et al.
algorithm. An unknown region in the image is filled using texture of
surrounding region. This implementation updates 1 patch at a time.

Outline of the algorithm for Texture Synthesis is as follows:
- Loop: Generate the boundary pixels of the region to be inpainted
    - Loop: Compute the priority of each pixel
        - Generate a template of (window, window), center: boundary pixel
        - confidence_term: avg amount of reliable information in template
        - data_term: strength of the isophote hitting this boundary pixel
        - priority = data_term * confidence_term
    - Repeat for all boundary pixels and chose the pixel with max priority
    - Template matching of the pixel with max priority
        - Generate a template of (window, window) around this pixel
        - Compute the Sum of Squared Difference (SSD) between template and
          similar sized patches across the image
        - Find the pixel with smallest SSD, such that patch isn't where
          template is located (False positive)
        - Update the intensity value of the unknown region of template as
          the corresponding value from matched patch
- Repeat until all pixels are inpainted

For further information refer to [1]_.

References
----------
.. [1] Criminisi, Antonio; Perez, P.; Toyama, K., "Region filling and object
       removal by exemplar-based image inpainting," Image Processing, IEEE
       Transactions on , vol.13, no.9, pp.1200,1212, Sept. 2004 doi: 10.
       1109/TIP.2004.833105.

"""
import numpy as np
from skimage.filter.inpaint_exemplar import inpaint_criminisi
import matplotlib.pyplot as plt
from skimage.data import checkerboard


image = checkerboard().astype(np.uint8)
mask = np.zeros_like(image, dtype=np.uint8)
paint_region = (slice(75, 125), slice(75, 125))

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
