""" 
======================= 
Morphological Filtering 
=======================

Morphological image processing is a  collection of non-linear operations related
to  the  shape or  morphology  of  features in  an  image,  such as  boundaries,
skeletons, etc. In  any given  technique, we probe an image with a small shape  
or template called structuring element. This helps to define the region of 
interest or neighborhood. Neighborhood of a pixel is defined as all the pixels 
with a value 1 in the structuring element. The structuring element is positioned
at all possible locations in the image and it is compared  with the 
corresponding neighbourhood of pixels. 

In this document we outline the following basic morphological operations:

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

1. http://www.mathworks.in/help/images/morphology-fundamentals-dilation-and-erosion.html#f18-14379
2. http://www.cs.auckland.ac.nz/courses/compsci773s1c/lectures/ImageProcessing-html/topic4.htm
3. http://en.wikipedia.org/wiki/Mathematical_morphology

General Instructions
====================
The following checks should be made for running the morphological functions:

**Image**:

* Type : numpy.ndarray 
* Data type : uint8 
* 2D

**Structuring Element**: 

* Type: Binary or boolean

.. note::
   Skimage supports NumPy data types and takes in images as type 'ndarray'.
   matplotlib.pyplot is a python library for providing MATLAB-like 
   functionality, hence the same function names. E.g: imshow 

Some quick functions to check the dimensions or type or the shape(size) of the 
image:

* ``type(image)``
* ``ndim(image)``
* ``image.shape``

Lets Get Started
================
Importing & displaying using ``io.imread()`` and ``io.imshow()``
----------------------------------------------------------------
``io.imread() has a the parameter 'as_grey=True' which ensures that the image is 
taken as a 2D rather than a 3D array with equal R,G,B values for a point, hence
no need of slicing.
"""
import matplotlib.pyplot as plt
import numpy as np
from skimage.data import data_dir
from skimage.util import img_as_ubyte
import skimage.io._io as io

phantom = img_as_ubyte(io.imread(data_dir+'/phantom.png', as_grey=True))
plt.imshow(phantom)
plt.show()
"""
.. image:: PLOT2RST.current_figure

EROSION
=======
Morphological ``erosion`` sets a pixel at (i,j) to the **minimum over all 
pixels in the neighborhood centered at (i,j)**. For defining the structuring 
element, we use disk(radius) function.
"""
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk

phantom = img_as_ubyte(io.imread(data_dir+'/phantom.png', as_grey=True))

selem = disk(6); 
eroded = erosion(phantom, selem)

fg, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(phantom)
ax1.set_title('Original')
ax2.imshow(eroded)
ax2.set_title('After Erosion')
plt.show()
"""
.. image:: PLOT2RST.current_figure

See how the white boundary of the image disappers or gets eroded
as we increse the size of the disk. Also notice the increase in size of the 
two black ellipses in the center and the disappearance of the 3-4 light grey
patches in the lower part of the image.

DILATION
========
Morphological ``dilation`` sets a pixel at (i,j) to the **maximum over all 
pixels in the neighborhood centered at (i,j)**. Dilation enlarges bright 
regions and shrinks dark regions.
"""
phantom = img_as_ubyte(io.imread(data_dir+'/phantom.png', as_grey=True))

selem = disk(6); 
dilate = dilation(phantom, selem)

fg, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(phantom)
ax1.set_title('Original')
ax2.imshow(dilate)
ax2.set_title('After Dilation')
plt.show()
"""
.. image:: PLOT2RST.current_figure

See how the white boundary of the image thickens or gets
dialted as we increse the size of the disk. Also notice the decrease in size
of the two black ellipses in the centre, with the thickening of the light grey
circle in the center and the 3-4 patches in the lower part of the image.

OPENING
=======
Morphological ``opening`` on an image is defined as an **erosion followed by a 
dilation**. Opening can remove small bright spots (i.e. "salt") and connect 
small dark cracks. 
"""
phantom = img_as_ubyte(io.imread(data_dir+'/phantom.png', as_grey=True))

selem = disk(6); 
opened = opening(phantom, selem)

fg, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(phantom)
ax1.set_title('Original')
ax2.imshow(opened)
ax2.set_title('After Opening')
plt.show()
"""
.. image:: PLOT2RST.current_figure

Since ``opening`` an image is equivalent to *erosion followed
by dilation*, white or lighter portions in the image which are smaller than the
structuring element tend to be removed, just as in erosion along with the
increase in thickness of black portions and thinning of larger (than structing
elements) white portions. But dilation reverses this effect and hence as we can
see in the image, the central 2 dark ellipses and the circular lighter portion
retain their thickness but the lighter patchs in the bottom get completely
eroded.

CLOSING
=======
Morphological ``closing`` on an image is defined as a **dilation followed by an 
erosion**. Closing can remove small dark spots (i.e. "pepper") and connect 
small bright cracks. 
"""
phantom = img_as_ubyte(io.imread(data_dir+'/phantom.png', as_grey=True)) 

selem = disk(6); 
closed = closing(phantom, selem)

fg, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(phantom)
ax1.set_title('Original')
ax2.imshow(closed)
ax2.set_title('After Closing')
plt.show()
"""
.. image:: PLOT2RST.current_figure

Comments : Since ``closing`` an image is equivalent to *dilation
followed by erosion*, the small black 10X10 pixel wide square introduced has
been removed and the -34 white ellipses at the bottom get connected, just as is
expected after dilation along with the thinning of larger (than structing
elements) black portions. But erosion reverses this effect and hence as we can
see in the image, the central 2 dark ellipses and the circular lighter portion
retain their thickness but the all black square is completely removed. But note
that the white patches at the bottom remain connected even after erosion.

WHITE TOPHAT
============
The ``white_tophat`` of an image is defined as the **image minus its 
morphological opening**. This operation returns the bright spots of the image 
that are smaller than the structuring element. 
"""
phantom = img_as_ubyte(io.imread(data_dir+'/phantom.png', as_grey=True)) 

selem = disk(6); 
w_tophat = white_tophat(phantom, selem)

fg, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(phantom)
ax1.set_title('Original')
ax2.imshow(w_tophat)
ax2.set_title('After performing white_tophat')
plt.show()
"""
.. image:: PLOT2RST.current_figure

This technique is used to locate the bright spots in an
image which are smaller than the size of the structuring element. As can be
seen below, the 10X10 pixel wide white square and a part of the white boundary 
are highlighted since they are smaller in size as compared to the disk which 
is of radius 5, i.e. 10 pixels wide. If the radius is decreased to 4, we can 
see that a center of the square is removed and only the corners are visible, 
since diagonals are longer than sides.

BLACK TOPHAT
============
The ``black_tophat`` of an image is defined as its morphological **closing minus 
the original image**. This operation returns the *dark spots of the image that
are smaller than the structuring element*. 
"""
phantom = img_as_ubyte(io.imread(data_dir+'/phantom.png', as_grey=True)) 
phantom[340:360, 200:220], phantom[100:110, 200:210] = 0, 0

selem = disk(6); 
b_tophat = black_tophat(phantom, selem)

fg, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(phantom)
ax1.set_title('Original')
ax2.imshow(b_tophat)
ax2.set_title('After Black Tophat')
plt.show()
"""
.. image:: PLOT2RST.current_figure

This technique is used to locate the dark spots in an image which are 
smaller than the size of the structuring element. As can be seen  below, the 
10X10 pixel wide black square is highlighted since it is smaller or equal in 
size as compared to the disk which is of radius 5, i.e. 10 pixels wide. If the 
radius is decreased to 4, we can see that a center of the square is removed and 
only the corners are visible, since diagonals are longer than sides.

Duality 
-------
In the sense that erosion tends to shrink the size of white objects while 
increasing the size of black objects. Conversely, dilation does just the 
opposite. Similarly, opening tends to eliminate black objects smaller than the
structuring element, wheres closing eliminates white objects.

1. Erosion <-> Dilation 
2. Opening <-> Closing 
3. White Tophat <-> Black Tophat

SKELETONIZE
===========
Thinning is used to reduce each connected component in a binary image to a 
**single-pixel wide skeleton**. It is important to note that this is performed
on binary images only.

"""
text = img_as_ubyte(io.imread(data_dir+'/ip_text.gif', as_grey=True)) 
text = text.astype(bool)

sk = skeletonize(text)

fg, (ax1, ax2) = plt.subplots(ncols=2)
ax1.imshow(text, vmin=0, vmax=1)
ax1.set_title('Original')
ax2.imshow(sk, vmin=0, vmax=1)
ax2.set_title('After Skeletonization')
plt.show()
"""
.. image:: PLOT2RST.current_figure

As the name suggests, this technique is used to thin the
image to 1-pixel wide skeleton by applying thinning successively.

CONVEX HULL
===========
The ``convex_hull_image`` is the **set of pixels included in the smallest 
convex polygon that surround all white pixels in the input image**. Again note 
that this is also performed on binary images.

"""
rooster = img_as_ubyte(io.imread(data_dir+'/rooster.png', as_grey=True))
rooster = rooster.astype(bool)

hull1 = convex_hull_image(rooster)
rooster1 = np.copy(rooster)
rooster1[350:355, 90:95] = 1
hull2 = convex_hull_image(rooster1)

fg, ax = plt.subplots(nrows=2, ncols=2)
ax[0, 0].imshow(rooster)
ax1.set_title('Original')
ax[0, 1].imshow(rooster1)
ax2.set_title('After adding a small grain')
ax[1, 0].imshow(hull1)
ax1.set_title('Convex Hull for Original')
ax[1, 1].imshow(hull2)
ax2.set_title('Convex Hull after adding the small grain')
plt.show()
"""
.. image:: PLOT2RST.current_figure

As the figure illustrates, convex_hull_image() gives the
smallestpolygon which covers the white or True completely in the image.

"""
