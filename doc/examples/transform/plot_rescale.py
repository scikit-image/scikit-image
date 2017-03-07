"""
=======
Rescale
=======

The Rescale operation performs interpolation to 
upscale or downscale images by a certain decimal 
factor.

"""


import matplotlib.pyplot as plt

from skimage import data
from skimage.transform import rescale

image = data.camera()

rescale_image = rescale(image, 0.5)

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3),
                               sharex=True, sharey=True,
                               subplot_kw={'adjustable':'box-forced'})

ax0.imshow(image, cmap=plt.cm.gray, interpolation='none')
ax0.axis('off')
ax1.imshow(rescale_image, cmap=plt.cm.gray, interpolation='none')
ax1.axis('off')

plt.show()
