"""
================
Blob Detection
================

Detect blobs in an image with the Difference of Gaussian (DoG) method and
plot their position on the image.

.. [1] http://en.wikipedia.org/wiki/Blob_detection#The_difference_of_Gaussians_approach

"""
from skimage.feature import blob_dog
from skimage import data, color
from matplotlib import pyplot as plt
import math

print """
Between December 18 - December 28, 1995 the Hubble Space telescopse took a
series of photographs of a seemingly empty empty patch which covered about
one 24-millionth of the whole sky. The results were astonishing.The example
used about 1/4th of the original picture.

Source
------
en.wikipedia.org/wiki/Hubble_Deep_Field
"""

img = data.hubble_deep_field()[0:300, 0:300]
gray_img = color.rgb2gray(img)
blobs = blob_dog(gray_img, threshold=.5, min_sigma=0.1, max_sigma=10)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.imshow(img, cmap='gray')
num = len(blobs)

for b in blobs:
    y, x = b[0], b[1]
    r = math.sqrt(b[2] / math.pi)
    c = plt.Circle((x, y), r, color='#ff0000', lw=2, fill=False)
    ax.add_patch(c)

print """This picture contains atleast %d detected stellar objects
Yet another reminder of the extraordinaty vastness of our universe.
""" % blobs.shape[0]

plt.show()
