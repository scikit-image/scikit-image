"""
========================
Histogram Backprojection
========================

Histogram Backprojection is a tool for image segmentation, object tracking
etc. based on object's color distribution..

To make it simpler, it calculates the histogram of an object and uses this
histogram as a feature to find the object in other images. So it takes two
images as input, one is your object image and next is the image where you
want to find the object. The function then calculates histograms of two
images and gives you a **backprojected image**. Backprojected image is a
grayscale image where each pixel denotes the probability of that pixel being
part of your object. Higher value means it is most likely part of your object.
So thresholding this backprojected image for a suitable value gives the mask
of your object. So for best results, your object should fill the object image
as far as possible.

In below example, the brown region in the image is needed to be segmented.
So, first 200x200 block of image is selected as object image. Then
backprojection is applied to it. The result is thresholded with Otsu's
thresholding to get the mask. Then a bitwise_and operation with input image
and mask gives the segmented object.

.. [1] Swain, Michael J., and Dana H. Ballard. "Indexing via color histograms."
Active Perception and Robot Vision. Springer Berlin Heidelberg, 1992. 261-273.

"""
import numpy as np
import matplotlib.pyplot as plt

from skimage.exposure import histogram_backproject
from skimage import data
from skimage.filter import threshold_otsu


img1 = data.immunohistochemistry()
img2 = img1[:200, :200]

# apply histogram backprojection
bc = histogram_backproject(img1, img2)

# threshold the image with otsu's thresholding
thresh = threshold_otsu(bc)
mask = np.where(bc >= thresh, 255, 0).astype('uint8')
mask = np.dstack((mask, mask, mask))
res = np.bitwise_and(img1, mask)

plt.subplot(131), plt.imshow(img1), plt.title('Input image')
plt.subplot(132), plt.imshow(bc, 'gray'), plt.title('Backprojected Image')
plt.subplot(133), plt.imshow(res), plt.title('Segmented Image')
plt.show()
