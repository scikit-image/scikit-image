"""
===============
Polar transform
===============

Polar transform is a transformation that converts polar coordinates to
cartesian coordinates. This transform is particularly interesting for images
that present a rotational symmetry.

"""

from skimage import data
from skimage.transform import cart2pol

image = data.retina()
center = (int(image.shape[0]/2), int(image.shape[1]/2))
polar_image = cart2pol(image, center=center, full_output=False)

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 4))

img0 = ax0.imshow(image)
ax0.set_title('Original image')

img1 = ax1.imshow(polar_image, aspect='auto')
ax1.set_title('Polar transform')
ax1.set_xlabel('Angle (Degree)')
ax1.set_ylabel('Radius')

plt.show()
