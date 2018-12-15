"""
========================
Histogram Backprojection
========================

Histogram Backprojection is a tool for image segmentation based on object's
color or intensity distribution.

The histogram of an object, which you want to detect or track, is used as a
feature to find the object in other images. So it takes two images as input,
one is your object image and the next is the image where you want to find the
object. The function then computes the histograms of these two images and
return a **backprojected image** [1]_. A backprojected image is a grayscale image
where each pixel denotes the probability of that pixel being part of the
object. By thresholding this backprojected image with a suitable value gives
the objects' mask.

In below example, the brown region in the image is needed to be segmented.
So, first 200x200 block of image is selected as object image. Then
backprojection is applied to it. The result is thresholded with Otsu's
thresholding to get the mask. Then a bitwise_and operation with input image
and mask gives the segmented object.


References
----------

.. [1] Swain, Michael J., and Dana H. Ballard. "Indexing via color histograms."
       Active Perception and Robot Vision. Springer Berlin Heidelberg,
       1992. 261-273.  :DOI:`10.1109/ICCV.1990.139558`

"""
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import histogram_backprojection
from skimage import data
from skimage.filters import threshold_otsu
from skimage import img_as_ubyte


image = data.immunohistochemistry()
image = img_as_ubyte(image)
template = image[50:200, :150]

# apply histogram backprojection
backprojection = histogram_backprojection(image, template)

# threshold the image with otsu's thresholding
thresh = threshold_otsu(backprojection)
mask = backprojection >= thresh
# Set zero values where the mask is False
res = np.where(mask[..., None], image, 0)

fig, ax = plt.subplots(nrows=2, ncols=2)
ax = ax.ravel()

ax[0].imshow(image)
ax[0].axis('off')
ax[0].set_title('Input image')

ax[1].imshow(template)
ax[1].axis('off')
ax[1].set_title('Template')

ax[2].imshow(backprojection, cmap=plt.cm.gray)
ax[2].axis('off')
ax[2].set_title('Backprojected image')

ax[3].imshow(res)
ax[3].axis('off')
ax[3].set_title('Segmented image')

plt.show()
