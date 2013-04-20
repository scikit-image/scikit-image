# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

"""
=======================
Morphological Filtering
=======================

Morphological image processing is a collection of non-linear operations related to the shape or morphology of features in an image, such as boundaries, skeletons, etc. Morphological operations rely only on the relative ordering of pixel values, not on their numerical values, and therefore are especially suited to the processing of binary images. It is based on set theory and makes use of logical operations to carry out the functionality and hence are less computationally intensive and easy to implement. In any given technique, we probe an image with a small shape or template called a structuring element. The structuring element is positioned at all possible locations in the image and it is compared with the corresponding neighbourhood of pixels. Neighborhood of a pixel is defined as all the pixels with a value 1 in the structuring element.

In this document we outline the following basic morphological operations :

1. Erosion
2. Dilation
3. Opening
4. Closing
5. White Tophat
6. Black Tophat
7. Skeletonize
8. Convex Hull

Additional Resources :
----------------------

1. Provides a nice understanding of the most basic operations involved in morphological processing, i.e. erosion and dilation :  http://www.mathworks.in/help/images/morphology-fundamentals-dilation-and-erosion.html#f18-14379
2. http://www.cs.auckland.ac.nz/courses/compsci773s1c/lectures/ImageProcessing-html/topic4.htm
3. For an insight into the mathematical algorithms for each function : http://en.wikipedia.org/wiki/Mathematical_morphology

Importing images
================
"""

# We will be looking at the following techniques to begin with
from skimage.morphology import erosion, dilation, opening, closing, white_tophat, black_tophat, skeletonize, convex_hull, convex_hull_image

# Importing 'square' and 'disk' modules for creating BINARY structuring elements
from skimage.morphology import square as sq
from skimage.morphology import disk

# Skimage supports NumPy data types and takes in images as type 'ndarray'. matplotlib.pyplot is a python library for providing MATLAB-like functionality, hence the same function names. E.g: imshow
import matplotlib.pyplot as plt
import numpy as np

# Importing the '_io' module for reading, writing and showing images. Note thatin skimage, all files having the same name as the folder have been renamed with an '_'. Hence '_io'
import skimage.io._io as io
from skimage.data import load, data_dir
from skimage.util import img_as_ubyte

# Importing images

# <markdowncell>

# There are essentially 2 ways for importing images for use with skimage. They are as follows :
# 
# 1. Using the plt.imread() to read the images as a *numpy.ndarray* data type
# 2. Using the skimage modules like 'imread', 'imshow' etc provided under the '/skimage/io/_io.py'
# 
# Note that plt.imread() will read the image, a grey-scale image as a 3D array, whereas with io.imread() there is a parameter 'as_grey=True' for reading it as a 2D array and using 'img_as_ubyte()', can be converted to type : *numpy.ndarray* with *uint8* element. This is the input array for the morphological functions.
# 
# The following checks should be made for running the morphological functions :
# 
# Image :
# 
# 1. Type : numpy.ndarray
# 2. Data type : uint8
# 3. 2D
# 
# Structuring Element :
# 
# 1. Type : Binary or boolean

# <headingcell level=3>

# 1. Importing & displaying using plt.imread() and plt.imshow()

# <codecell>

i_f = plt.imread('/Users/chintak/Repositories/scikit-image/skimage/data/bw_text.png')
# Type : numpy.ndarray, dtype : float32, dimensions : 3
#type(i_f)
#i_f
#ndim(i_f)
# Slicing
i_f2 = i_f[:,:,0]
# To convert to uint8 data type
i = img_as_ubyte(i_f2)

# To display this image
plt.imshow(i, cmap=plt.cm.gray)  # For showing gray-scale images
#ndim(i)
plt.show()

# <headingcell level=3>

# 2. Importing & displaying using io.imread() and io.imshow()

# <codecell>

phantom = img_as_ubyte(io.imread('/Users/chintak/Repositories/scikit-image/skimage/data/phantom.png', as_grey=True))
# 'as_grey=True' ensures that the image is taken as a 2D rather than a 3D array with equal R,G,B values for a point
io.imshow(phantom)
plt.show()

# <headingcell level=2>

# EROSION

# <markdowncell>

# Usage : erosion(image, selem, out=None, shift_x=False, shift_y=False)
# 
# Return greyscale morphological erosion of an image.
# 
# Morphological erosion sets a pixel at (i,j) to the **minimum over all pixels
# in the neighborhood centered at (i,j)**. Erosion shrinks bright regions and
# enlarges dark regions.
# 
# Parameters
# ----------
# image : ndarray
#     Image array.
# selem : ndarray
#     The neighborhood expressed as a 2-D array of 1's and 0's.
# out : ndarray
#     The array to store the result of the morphology. If None is
#     passed, a new array will be allocated.
# shift_x, shift_y : bool
#     shift structuring element about center point. This only affects
#     eccentric structuring elements (i.e. selem with even numbered sides).
# 
# Returns
# -------
# eroded : uint8 array
#     The result of the morphological erosion.

# <codecell>

# We will be working with phantom.png for this function.
# First defining the structuring element as a disk using disk()
#selem = disk(3);
selem = disk(6);
#selem = disk(10);
eroded = erosion(phantom, selem)

# Displaying the original and eroded image 
# 'plt.figure() can be used for showing multiple images together
plt.figure(1) 
io.imshow(phantom)
plt.figure(2)
io.imshow(eroded)
plt.show()

# <markdowncell>

# **Comments** : 
# 
# See how the white boundary of the image disappers or gets eroded as we increse the size of the disk. 
# Also notice the increase in size of the two black ellipses in the center and the disappearance of the 3-4 light grey patches in the lower part of the image.

# <headingcell level=2>

# DILATION

# <markdowncell>

# Documentation :
# 
# Definition: dilation(image, selem, out=None, shift_x=False, shift_y=False)
# Docstring:
# Return greyscale morphological dilation of an image.
# 
# Morphological dilation sets a pixel at (i,j) to the **maximum over all pixels
# in the neighborhood centered at (i,j)**. Dilation enlarges bright regions
# and shrinks dark regions.
# 
# Parameters
# ----------
# 
# image : ndarray
#     Image array.
# selem : ndarray
#     The neighborhood expressed as a 2-D array of 1's and 0's.
# out : ndarray
#     The array to store the result of the morphology. If None, is
#     passed, a new array will be allocated.
# shift_x, shift_y : bool
#     shift structuring element about center point. This only affects
#     eccentric structuring elements (i.e. selem with even numbered sides).
# 
# Returns
# -------
# dilated : uint8 array
#     The result of the morphological dilation.

# <codecell>

# We will be working with phantom.png to show difference between erosion and dilation.
# First defining the structuring element as a disk using disk()
#selem = disk(3);
selem = disk(6);
#selem = disk(10);
dilate = dilation(phantom, selem)

# Displaying the original and eroded image 
# 'plt.figure() can be used for showing multiple images together
plt.figure(1) 
io.imshow(phantom)
plt.figure(2)
io.imshow(dilate)
plt.show()

# <markdowncell>

# **Comments** : 
# 
# See how the white boundary of the image thickens or gets dialted as we increse the size of the disk. 
# Also notice the decrease in size of the two black ellipses in the centre, with the thickening of the light grey circle in the center and the 3-4 patches in the lower part of the image.

# <headingcell level=2>

# OPENING

# <markdowncell>

# Documentation :
# 
# Definition: opening(image, selem, out=None)
# Docstring:
# Return greyscale morphological opening of an image.
# 
# The morphological opening on an image is defined as an **erosion followed by
# a dilation**. Opening can remove small bright spots (i.e. "salt") and connect
# small dark cracks. This tends to "open" up (dark) gaps between (bright)
# features.
# 
# Parameters
# ----------
# image : ndarray
#     Image array.
# selem : ndarray
#     The neighborhood expressed as a 2-D array of 1's and 0's.
# out : ndarray
#     The array to store the result of the morphology. If None
#     is passed, a new array will be allocated.
# 
# Returns
# -------
# opening : uint8 array
#     The result of the morphological opening.

# <codecell>

# We will be working with phantom.png for this function.
# First defining the structuring element as a disk using disk()
#selem = disk(3);
selem = disk(6);
#selem = disk(10);
opened = opening(phantom, selem)

# Displaying the original and eroded image 
# 'plt.figure() can be used for showing multiple images together
plt.figure(1) 
io.imshow(phantom)
plt.figure(2)
io.imshow(opened)
plt.show()

# <markdowncell>

# **Comments** : 
# 
# Since 'opening' an image is equivalent to *erosion followed by dilation*, white or lighter portions in the image which are smaller than the structuring element tend to be removed, just as in erosion along with the increase in thickness of black portions and thinning of larger (than structing elements) white portions. But dilation reverses this effect and hence as we can see in the image, the central 2 dark ellipses and the circular lighter portion retain their thickness but the lighter patchs in the bottom get completely eroded.

# <headingcell level=2>

# CLOSING

# <markdowncell>

# Documentation :
# 
# Definition: closing(image, selem, out=None)
# Docstring:
# Return greyscale morphological closing of an image.
# 
# The morphological closing on an image is defined as a **dilation followed by
# an erosion**. Closing can remove small dark spots (i.e. "pepper") and connect
# small bright cracks. This tends to "close" up (dark) gaps between (bright)
# features.
# 
# Parameters
# ----------
# image : ndarray
#     Image array.
# selem : ndarray
#     The neighborhood expressed as a 2-D array of 1's and 0's.
# out : ndarray
#     The array to store the result of the morphology. If None,
#     is passed, a new array will be allocated.
# 
# Returns
# -------
# closing : uint8 array
#     The result of the morphological closing.

# <codecell>

# We will be working with phantom.png for this function.
# First defining the structuring element as a disk using disk()
#selem = disk(3);
selem = disk(6);
#selem = disk(10);
phantom1 = phantom
phantom[300:310, 200:210] = 0
closed = closing(phantom1, selem)

# Displaying the original and eroded image 
# 'plt.figure() can be used for showing multiple images together
plt.figure(1) 
io.imshow(phantom1)
plt.figure(2)
io.imshow(closed)
plt.show()

# <markdowncell>

# **Comments** : 
# 
# Since 'closing' an image is equivalent to *dilation followed by erosion*, the small black 10X10 pixel wide square introduced has been removed and the -34 white ellipses at the bottom get connected, just as is expected after dilation along with the thinning of larger (than structing elements) black portions. But erosion reverses this effect and hence as we can see in the image, the central 2 dark ellipses and the circular lighter portion retain their thickness but the all black square is completely removed. But note that the white patches at the bottom remain connected even after erosion.

# <headingcell level=2>

# WHITE TOPHAT

# <markdowncell>

# Documentation :
# 
# Definition: white_tophat(image, selem, out=None)
# Docstring:
# Return white top hat of an image.
# 
# The white top hat of an image is defined as the **image minus its
# morphological opening**. This operation returns the bright spots of the image
# that are smaller than the structuring element.
# 
# Parameters
# ----------
# image : ndarray
#     Image array.
# selem : ndarray
#     The neighborhood expressed as a 2-D array of 1's and 0's.
# out : ndarray
#     The array to store the result of the morphology. If None
#     is passed, a new array will be allocated.
# 
# Returns
# -------
# opening : uint8 array
#     The result of the morphological white top hat.

# <codecell>

# We will be working with phantom.png for this function.
# First defining the structuring element as a disk using disk()
#selem = disk(3);
selem = disk(5);
#selem = disk(10);
phantom[150:160, 200:210] = 255
w_tophat = white_tophat(phantom, selem)

# Displaying the original and eroded image 
# 'plt.figure() can be used for showing multiple images together
plt.figure(1) 
io.imshow(phantom)
plt.figure(2)
io.imshow(w_tophat)
#plt.figure(3)
#io.imshow(opening(phantom, selem))
plt.show()

# <markdowncell>

# **Comments** : 
# 
# This technique is used to locate the bright spots in an image which are smaller than the size of the structuring element. As can be seen, the 10X10 pixel wide white square and a part of the white boundary are highlighted since they are smaller in size as compared to the disk which is of radius 5, i.e. 10 pixels wide. If the radius is decreased to 4, we can see that a center of the square is removed and only the corners are visible, since diagonals are longer than sides.

# <headingcell level=2>

# BLACK TOPHAT

# <markdowncell>

# Documentation :
# 
# Definition: black_tophat(image, selem, out=None)
# Docstring:
# Return black top hat of an image.
# 
# The black top hat of an image is defined as its morphological **closing minus
# the original image**. This operation returns the *dark spots of the image that
# are smaller than the structuring element*. Note that dark spots in the
# original image are bright spots after the black top hat.
# 
# Parameters
# ----------
# image : ndarray
#     Image array.
# selem : ndarray
#     The neighborhood expressed as a 2-D array of 1's and 0's.
# out : ndarray
#     The array to store the result of the morphology. If None
#     is passed, a new array will be allocated.
# 
# Returns
# -------
# opening : uint8 array
#    The result of the black top filter.

# <codecell>

# We will be working with phantom.png for this function.
# First defining the structuring element as a disk using disk()
#selem = disk(3);
selem = disk(5);
#selem = disk(10);
phantom[150:160, 200:210] = 255
b_tophat = black_tophat(phantom, selem)

# Displaying the original and eroded image 
# 'plt.figure() can be used for showing multiple images together
plt.figure(1) 
io.imshow(phantom)
plt.figure(2)
io.imshow(b_tophat)
#plt.figure(3)
#io.imshow(opening(phantom, selem))
plt.show()

# <markdowncell>

# **Comments** : 
# 
# This technique is used to locate the dark spots in an image which are smaller than the size of the structuring element. As can be seen, the 10X10 pixel wide black square is highlighted since it is smaller or equal in size as compared to the disk which is of radius 5, i.e. 10 pixels wide. If the radius is decreased to 4, we can see that a center of the square is removed and only the corners are visible, since diagonals are longer than sides.

# <markdowncell>

# Duality
# -------
# 
# 1. Erosion <-> Dilation
# 2. Opening <-> Closing
# 3. White Tophat <-> Black Tophat

# <headingcell level=2>

# SKELETONIZE

# <markdowncell>

# Documentation :
# 
# Definition: skeletonize(image)
# Docstring:
# Return the skeleton of a **binary image**.
# 
# Thinning is used to reduce each connected component in a binary image
# to a **single-pixel wide skeleton**.
# 
# Parameters
# ----------
# image : numpy.ndarray
#     A binary image containing the objects to be skeletonized. '1'
#     represents foreground, and '0' represents background. It
#     also accepts arrays of boolean values where True is foreground.
# 
# Returns
# -------
# skeleton : ndarray
#     A matrix containing the thinned image.
# 
# See also
# --------
# medial_axis
# 
# Notes
# -----
# The algorithm [1] works by making successive passes of the image,
# removing pixels on object borders. This continues until no
# more pixels can be removed.  The image is correlated with a
# mask that assigns each pixel a number in the range [0...255]
# corresponding to each possible pattern of its 8 neighbouring
# pixels. A look up table is then used to assign the pixels a
# value of 0, 1, 2 or 3, which are selectively removed during
# the iterations.
# 
# Note that this algorithm will give different results than a
# medial axis transform, which is also often referred to as
# "skeletonization".
# 
# References
# ----------
# .. [1] A fast parallel algorithm for thinning digital patterns,
#    T. Y. ZHANG and C. Y. SUEN, Communications of the ACM,
#    March 1984, Volume 27, Number 3

# <codecell>

# For this we will use ip_text.gif
text = img_as_ubyte(io.imread('/Users/chintak/Repositories/scikit-image/skimage/data/ip_text.gif', as_grey=True)).astype(bool)
skeleton = skeletonize(text)

# Displaying the original image and the skeletonized image
plt.figure(1)
io.imshow(text)
plt.figure(2)
io.imshow(skeleton)
plt.show()

# <markdowncell>

# **Comments** : 
# 
# As the name suggests, this technique is used to thin the image to 1-pixel wide skeleton by applying thinning successively.

# <headingcell level=2>

# CONVEX HULL

# <markdowncell>

# Documentation :
# 
# Definition: convex_hull_image(image)
# Docstring:
# Compute the convex hull image of a **binary image**.
# 
# The convex hull is the **set of pixels included in the smallest convex
# polygon that surround all white pixels in the input image**.
# 
# Parameters
# ----------
# image : ndarray
#     Binary input image.  This array is cast to bool before processing.
# 
# Returns
# -------
# hull : ndarray of uint8
#     Binary image with pixels in convex hull set to 255.
# 
# References
# ----------
# .. [1] http://blogs.mathworks.com/steve/2011/10/04/binary-image-convex-hull-algorithm-notes/

# <codecell>

# For this we will use chicken.png
rooster = img_as_ubyte(io.imread('/Users/chintak/Repositories/scikit-image/skimage/data/rooster.png', as_grey=True)).astype(bool)
hull1 = convex_hull_image(rooster)
rooster1 = np.copy(rooster)
rooster1[350:355, 90:95] = 1
hull2 = convex_hull_image(rooster1)

# Displaying the original image and the skeletonized image
plt.figure(1)
io.imshow(rooster)
plt.figure(2)
io.imshow(rooster1)
plt.figure(3)
io.imshow(hull1)
plt.figure(4)
io.imshow(hull2)
plt.show()

