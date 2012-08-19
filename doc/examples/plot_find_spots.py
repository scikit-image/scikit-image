"""
==========
Find spots
==========

In this example, we find bright spots in an image using morphological
reconstruction by dilation. Dilation expands the maximal values of the seed
image until it encounters a mask image. Thus, the seed image and mask image
represent the minimum and maximum possible values of the reconstructed image.

We start with an image containing both peaks and holes:
"""
import matplotlib.pyplot as plt

from skimage import data
from skimage.exposure import rescale_intensity

image = data.moon()
# Rescale image intensity so that we can see dim features.
image = rescale_intensity(image, in_range=(50, 200))

# convenience function for plotting images
def imshow(image, **kwargs):
    plt.figure(figsize=(5, 4))
    plt.imshow(image, **kwargs)
    plt.axis('off')

imshow(image)
plt.title('original image')

"""
.. image:: PLOT2RST.current_figure

Now we need to create the seed image, where the maxima represent the starting
points for dilation. To find spots, we initialize the seed image to the minimum
value of the original image. Along the borders, however, we use the original
values of the image. These border pixels will be the starting points for the
dilation process. We then limit the dilation by setting the mask to the values
of the original image.
"""

import numpy as np
from skimage.morphology import reconstruction

seed = np.copy(image)
seed[1:-1, 1:-1] = image.min()
mask = image

rec = reconstruction(seed, mask, method='dilation')

imshow(rec, vmin=image.min(), vmax=image.max())
plt.title('')

"""
.. image:: PLOT2RST.current_figure

As shown above, dilating inward from the edges removes peaks, since (by
definition) peaks are surrounded by pixels of darker value. Finally, we can
isolate the bright spots by subtracting the reconstructed image from the
original image.
"""

imshow(image - rec)
plt.title('"holes"')
plt.show()

"""
.. image:: PLOT2RST.current_figure
"""
