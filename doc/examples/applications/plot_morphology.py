""" 
======================= 
Morphological Filtering 
=======================

Morphological image processing is a  collection of non-linear operations related
to  the  shape or  morphology  of  features in  an  image,  such as  boundaries,
skeletons, etc. Morphological  operations rely only on the  relative ordering of
pixel values, not on their numerical values, and therefore are especially suited
to the processing of  binary images. It is based on set theory  and makes use of
logical  operations  to   carry  out  the  functionality  and   hence  are  less
computationally  intensive and  easy to  implement. In  any given  technique, we
probe an image with a small shape  or template called a structuring element. The
structuring element is positioned at all  possible locations in the image and it
is compared  with the corresponding  neighbourhood of pixels. Neighborhood  of a
pixel is defined  as all the pixels  with a value 1 in  the structuring element.

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

.. [1] Provides a nice understanding of the most basic operations involved in
       morphological processing, i.e. erosion and dilation
       `Link text <http://goo.gl/Cs4n6>`_
.. [2] Auckland university: `Link text <http://goo.gl/Ylf19>`_
.. [3] For an insight into the mathematical algorithms for each function:
       `Link text <http://en.wikipedia.org/wiki/Mathematical_morphology>`_

Importing images 
================ 
There are essentially 2 ways for importing images for use with skimage. They are
as  follows  :  

* Using the plt.imread()  to read the images as a  *numpy.ndarray* data type 
* Using  the  skimage modules  like  'imread',  'imshow'  etc provided  under  
  the '/skimage/io/_io.py'

.. Note:: 
   plt.imread() will read the image, a grey-scale image as a 3D array, whereas 
   with io.imread() there is a parameter 'as_grey=True' for reading it as a2D 
   array and using 'img_as_ubyte()', can be converted to type : *numpy.ndarray* 
   with *uint8* element. This is the input array for the morphological function

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

* type(image)
* ndim(image)
* image.shape

Importing & displaying using plt.imread() and plt.imshow()
-----------------
"""
import matplotlib.pyplot as plt
import numpy as np
from skimage.data import data_dir
from skimage.util.dtype import dtype_range, convert

i_f = plt.imread(data_dir+'/phantom.png')
i_f2 = i_f[:,:,0] 
i = convert(i_f2, np.uint8)
plt.imshow(i, cmap=plt.cm.gray, vmin=0, vmax=255)
plt.show()

"""
.. image:: PLOT2RST.current_figure

Importing & displaying using io.imread() and io.imshow()
-------------------
The advantage of using 'as_grey=True' is that it ensures that the image is 
taken as a 2D rather than a 3D array with equal R,G,B values for a point, hence
no need of slicing.
"""
import matplotlib.pyplot as plt
import numpy as np
import skimage.io._io as io
from skimage.data import data_dir
from skimage.util.dtype import dtype_range, convert

phantom = convert(io.imread(data_dir+'/phantom.png', as_grey=True), np.uint8) 
plt.imshow(phantom, cmap=plt.cm.gray, vmin=0, vmax=255)
plt.show()
"""
.. image:: PLOT2RST.current_figure

EROSION
=======
Morphological erosion sets a pixel at (i,j) to the **minimum over all pixels 
in the neighborhood centered at (i,j)**. For defining the structuring element,
we use disk(radius) function.
 
**Comments**: See how the white boundary of the image disappers or gets eroded
as we increse the size of the disk. # Also notice the increase in size of the 
two black ellipses in the center and the disappearance of the 3-4 light grey
patches in the lower part of the image.
"""
from skimage.morphology import erosion, disk
import matplotlib.pyplot as plt
import numpy as np
import skimage.io._io as io
from skimage.data import data_dir
from skimage.util.dtype import dtype_range, convert

phantom = convert(io.imread(data_dir+'/phantom.png', as_grey=True), np.uint8) 
plt.imshow(phantom, cmap=plt.cm.gray, vmin=0, vmax=255)

selem = disk(6); 
eroded = erosion(phantom, selem)

plt.figure(figsize=[10, 30]) 
plt.subplot(121)
plt.imshow(phantom, cmap=plt.cm.gray, vmin=0, vmax=255)
plt.title('Original')
plt.subplot(122)
plt.imshow(eroded, cmap=plt.cm.gray, vmin=0, vmax=255)
plt.title('After Erosion')
plt.show()
"""
.. image:: PLOT2RST.current_figure

DILATION
========
Morphological dilation sets a pixel at (i,j) to the **maximum over all pixels 
in the neighborhood centered at (i,j)**. Dilation enlarges bright regions and 
shrinks dark regions.

**Comments**: See how the white boundary of the image thickens or gets
dialted as we increse the size of the disk. # Also notice the decrease in size
of the two black ellipses in the centre, with the thickening of the light grey
circle in the center and the 3-4 patches in the lower part of the image.
"""
from skimage.morphology import dilation, disk
import matplotlib.pyplot as plt
import numpy as np
import skimage.io._io as io
from skimage.data import data_dir
from skimage.util.dtype import dtype_range, convert

phantom = convert(io.imread(data_dir+'/phantom.png', as_grey=True), np.uint8) 
plt.imshow(phantom, cmap=plt.cm.gray, vmin=0, vmax=255)

selem = disk(6); 
dilate = dilation(phantom, selem)

plt.figure(figsize=[10, 30]) 
plt.subplot(121)
plt.imshow(phantom, cmap=plt.cm.gray, vmin=0, vmax=255)
plt.title('Original')
plt.subplot(122)
plt.imshow(dilate, cmap=plt.cm.gray, vmin=0, vmax=255)
plt.title('After Dilation')
plt.show()
"""
.. image:: PLOT2RST.current_figure

OPENING
=======
Morphological opening on an image is defined as an **erosion followed by a 
dilation**. Opening can remove small bright spots (i.e. "salt") and connect 
small dark cracks. 

**Comments**: Since 'opening' an image is equivalent to *erosion followed
by dilation*, white or lighter portions in the image which are smaller than the
structuring element tend to be removed, just as in erosion along with the
increase in thickness of black portions and thinning of larger (than structing
elements) white portions. But dilation reverses this effect and hence as we can
see in the image, the central 2 dark ellipses and the circular lighter portion
retain their thickness but the lighter patchs in the bottom get completely
eroded.
"""
from skimage.morphology import opening, disk
import matplotlib.pyplot as plt
import numpy as np
import skimage.io._io as io
from skimage.data import data_dir
from skimage.util.dtype import dtype_range, convert

phantom = convert(io.imread(data_dir+'/phantom.png', as_grey=True), np.uint8) 
plt.imshow(phantom, cmap=plt.cm.gray, vmin=0, vmax=255)

selem = disk(6); 
opened = opening(phantom, selem)

plt.figure(figsize=[10, 30]) 
plt.subplot(121)
plt.imshow(phantom, cmap=plt.cm.gray, vmin=0, vmax=255)
plt.title('Original')
plt.subplot(122)
plt.imshow(opened, cmap=plt.cm.gray, vmin=0, vmax=255)
plt.title('After Opening')
plt.show()
"""
.. image:: PLOT2RST.current_figure

CLOSING
=======
Morphological closing on an image is defined as a **dilation followed by an 
erosion**. Closing can remove small dark spots (i.e. "pepper") and connect 
small bright cracks. 

**Comments** : Since 'closing' an image is equivalent to *dilation
followed by erosion*, the small black 10X10 pixel wide square introduced has
been removed and the -34 white ellipses at the bottom get connected, just as is
expected after dilation along with the thinning of larger (than structing
elements) black portions. But erosion reverses this effect and hence as we can
see in the image, the central 2 dark ellipses and the circular lighter portion
retain their thickness but the all black square is completely removed. But note
that the white patches at the bottom remain connected even after erosion.
"""
from skimage.morphology import closing, disk
import matplotlib.pyplot as plt
import numpy as np
import skimage.io._io as io
from skimage.data import data_dir
from skimage.util.dtype import dtype_range, convert

phantom = convert(io.imread(data_dir+'/phantom.png', as_grey=True), np.uint8) 
plt.imshow(phantom, cmap=plt.cm.gray, vmin=0, vmax=255)

selem = disk(6); 
closed = closing(phantom, selem)

plt.figure(figsize=[10, 30]) 
plt.subplot(121)
plt.imshow(phantom, cmap=plt.cm.gray, vmin=0, vmax=255)
plt.title('Original')
plt.subplot(122)
plt.imshow(closed, cmap=plt.cm.gray, vmin=0, vmax=255)
plt.title('After Closing')
plt.show()
"""
.. image:: PLOT2RST.current_figure

WHITE TOPHAT
============
The white top hat of an image is defined as the **image minus its morphological 
opening**. This operation returns the bright spots of the image that are smaller
than the structuring element. 

**Comments**: This technique is used to locate the bright spots in an
image which are smaller than the size of the structuring element. As can be
seen below, the 10X10 pixel wide white square and a part of the white boundary 
are highlighted since they are smaller in size as compared to the disk which 
is of radius 5, i.e. 10 pixels wide. If the radius is decreased to 4, we can see
that a center of the square is removed and only the corners are visible, since
diagonals are longer than sides.
"""
from skimage.morphology import white_tophat, disk
import matplotlib.pyplot as plt
import numpy as np
import skimage.io._io as io
from skimage.data import data_dir
from skimage.util.dtype import dtype_range, convert

phantom = convert(io.imread(data_dir+'/phantom.png', as_grey=True), np.uint8) 
plt.imshow(phantom, cmap=plt.cm.gray, vmin=0, vmax=255)

selem = disk(6); 
w_tophat = white_tophat(phantom, selem)

plt.figure(figsize=[10, 30]) 
plt.subplot(121)
plt.imshow(phantom, cmap=plt.cm.gray, vmin=0, vmax=255)
plt.title('Original')
plt.subplot(122)
plt.imshow(w_tophat, cmap=plt.cm.gray, vmin=0, vmax=255)
plt.title('After White Tophat')
plt.show()
"""
.. image:: PLOT2RST.current_figure

BLACK TOPHAT
============
The black top hat of an image is defined as its morphological **closing minus 
the original image**. This operation returns the *dark spots of the image that
are smaller than the structuring element*. 

**Comments**: This technique is used to locate the dark spots in an image
which are smaller than the size of the structuring element. As can be seen 
below, the
10X10 pixel wide black square is highlighted since it is smaller or equal in
size as compared to the disk which is of radius 5, i.e. 10 pixels wide. If the
radius is decreased to 4, we can see that a center of the square is removed and
only the corners are visible, since diagonals are longer than sides.
"""
from skimage.morphology import black_tophat, disk
import matplotlib.pyplot as plt
import numpy as np
import skimage.io._io as io
from skimage.data import data_dir
from skimage.util.dtype import dtype_range, convert

phantom = convert(io.imread(data_dir+'/phantom.png', as_grey=True), np.uint8) 
phantom[340:360, 200:220], phantom[100:110, 200:210] = 0, 0

selem = disk(6); 
b_tophat = black_tophat(phantom, selem)

plt.figure(figsize=[10, 30]) 
plt.subplot(121)
plt.imshow(phantom, cmap=plt.cm.gray)
plt.title('Original')
plt.subplot(122)
plt.imshow(b_tophat, cmap=plt.cm.gray)
plt.title('After Black Tophat')
plt.show()
"""
.. image:: PLOT2RST.current_figure

Duality 
-------
1. Erosion <-> Dilation 
2. Opening <-> Closing 
3. White Tophat <-> Black Tophat

SKELETONIZE
===========
Thinning is used to reduce each connected component in a binary image to a 
**single-pixel wide skeleton**. It is important to note that this is performed
on binary images only.

**Comments**: As the name suggests, this technique is used to thin the
image to 1-pixel wide skeleton by applying thinning successively.
"""
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import numpy as np
import skimage.io._io as io
from skimage.data import data_dir
from skimage.util.dtype import dtype_range, convert

text = convert(io.imread(data_dir+'/ip_text.gif', as_grey=True), np.uint8) 

text = text.astype(bool)
sk = skeletonize(text)

plt.figure(figsize=[10, 30]) 
plt.subplot(121)
plt.imshow(text, cmap=plt.cm.gray, vmin=0, vmax=1)
plt.title('Original')
plt.subplot(122)
plt.imshow(sk, cmap=plt.cm.gray, vmin=0, vmax=1)
plt.title('After Skeletonize')
plt.show()
"""
.. image:: PLOT2RST.current_figure

CONVEX HULL
===========
The convex hull is the **set of pixels included in the smallest convex polygon 
that surround all white pixels in the input image**. Again note that this is 
also performed on binary images.

**Comments**: As the figure illustrates, convex_hull_image() gives the
smallestpolygon which covers the white or True completely in the image.
"""
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
import numpy as np
import skimage.io._io as io
from skimage.data import data_dir
from skimage.util.dtype import dtype_range, convert

rooster = convert(io.imread(data_dir+'/rooster.png', as_grey=True), np.uint8)
rooster = rooster.astype(bool) 
hull1 = convex_hull_image(rooster)
rooster1 = np.copy(rooster)
rooster1[350:355, 90:95] = 1
hull2 = convex_hull_image(rooster1)

plt.figure(figsize=[10, 30]) 
plt.subplot(221)
plt.imshow(rooster, cmap=plt.cm.gray, vmin=0, vmax=1)
plt.title('Original')
plt.subplot(222)
plt.imshow(rooster1, cmap=plt.cm.gray, vmin=0, vmax=1)
plt.title('After adding a small "grain"')
plt.subplot(223)
plt.imshow(hull1, cmap=plt.cm.gray, vmin=0, vmax=1)
plt.title('After applying Convex Hull to Original')
plt.subplot(224)
plt.imshow(hull2, cmap=plt.cm.gray, vmin=0, vmax=1)
plt.title('After applying Convex Hull to modified image')
plt.show()
"""
.. image:: PLOT2RST.current_figure

"""
