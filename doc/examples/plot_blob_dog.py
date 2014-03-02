"""
================
Blob Detection
================

Detect blobs in an image with the Difference of Gaussian ( DoG ) method and
plot their position on the image.

.. [1] http://en.wikipedia.org/wiki/Blob_detection#The_difference_of_Gaussians_approach

"""
from skimage.feature import blob_dog
from skimage import data
from matplotlib import pyplot as plt
import math

img = data.coins()
blobs = blob_dog(img)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.imshow(img,cmap='gray')


for b in blobs:
    x,y = b[0],b[1]
    r = math.sqrt( b[2]/math.pi )
    c = plt.Circle((y,x),r,color='#ff0000',lw = 2,fill = False)
    ax.add_patch(c)

plt.show()

