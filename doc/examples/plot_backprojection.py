"""
========================
Histogram Backprojection
========================

Histogram Backprojection is a tool for image segmentation, object tracking
etc. based on object's color or intensity distribution.

In simple words, it calculates the histogram of an object(which you want to
detect or track) and uses this histogram as a feature to find the object in
other images. So it takes two images as input, one is your object image and
the next is the image where you want to find the object. The function then
calculates histograms of two images and gives you a **backprojected image**.
Backprojected image is a grayscale image where each pixel denotes the
probability of that pixel being part of your object. Higher value means it
is most likely part of your object. So thresholding this backprojected image
for a suitable value gives the mask of your object. So for best results,
your object should fill the object image as far as possible.

In below example, the brown region in the image is needed to be segmented.
So, first 200x200 block of image is selected as object image. Then
backprojection is applied to it. The result is thresholded with Otsu's
thresholding to get the mask. Then a bitwise_and operation with input image
and mask gives the segmented object.

.. [1] Swain, Michael J., and Dana H. Ballard. "Indexing via color histograms."
       Active Perception and Robot Vision. Springer Berlin Heidelberg,
       1992. 261-273.

"""
import numpy as np
import matplotlib.pyplot as plt

from skimage.exposure import histogram_backproject
from skimage import data
from skimage.filter import threshold_otsu
from skimage import img_as_ubyte


img1 = data.immunohistochemistry()
img1 = img_as_ubyte(img1)
img2 = img1[:200, :200]

# apply histogram backprojection
bc = histogram_backproject(img1, img2)

# threshold the image with otsu's thresholding
thresh = threshold_otsu(bc)
mask = np.where(bc >= thresh, 255, 0).astype('uint8')
res = np.bitwise_and(img1, mask[..., np.newaxis])

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

ax1.imshow(img1)
ax1.axis('off')
ax1.set_title('input image')

ax2.imshow(mask, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('backprojected image')

ax3.imshow(res)
ax3.axis('off')
ax3.set_title('segmented image')

plt.show()
