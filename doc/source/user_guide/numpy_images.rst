==================================
A crash course on NumPy for images
==================================

Images manipulated by ``scikit-image`` are simply NumPy arrays. Hence, a
large fraction of operations on images will just consist in using NumPy::

    >>> from skimage import data
    >>> camera = data.camera()
    >>> type(camera)
    <type 'numpy.ndarray'>

Retrieving the geometry of the image and the number of pixels: ::

    >>> camera.shape
    (512, 512)
    >>> camera.size
    262144

Retrieving statistical information about gray values: ::

    >>> camera.min(), camera.max()
    (0, 255)
    >>> camera.mean()
    118.31400299072266

NumPy arrays representing images can be of different integer of float
numerical types. See :ref:`data_types` for more information about these
types and how scikit-image treats them.


NumPy indexing
--------------

NumPy indexing can be used both for looking at pixel values, and to
modify pixel values: ::

    >>> # Get the value of the pixel on the 10th row and 20th column
    >>> camera[10, 20]
    153
    >>> # Set to black the pixel on the 3rd row and 10th column
    >>> camera[3, 10] = 0

Be careful: in NumPy indexing, the first dimension (``camera.shape[0]``)
corresponds to rows, while the second (``camera.shape[1]``) corresponds
to columns, with the origin (``camera[0, 0]``) on the top-left corner.
This matches matrix/linear algebra notation, but is in contrast to
Cartesian (x, y) coordinates. See `Coordinate conventions`_ below for
more details.

Beyond individual pixels, it is possible to access / modify values of
whole sets of pixels, using the different indexing possibilities of
NumPy.

Slicing::

    >>> # Set to black the ten first lines
    >>> camera[:10] = 0

Masking (indexing with masks of booleans)::

    >>> mask = camera < 87
    >>> # Set to "white" (255) pixels where mask is True
    >>> camera[mask] = 255

Fancy indexing (indexing with sets of indices) ::

    >>> inds_r = np.arange(len(camera))
    >>> inds_c = 4 * inds_r % len(camera)
    >>> camera[inds_r, inds_c] = 0

Using masks, especially, is very useful to select a set of pixels on
which to perform further manipulations. The mask can be any boolean array
of same shape as the image (or a shape broadcastable to the image shape).
This can be useful to define a region of interest, such as a
disk: ::

    >>> nrows, ncols = camera.shape
    >>> row, col = np.ogrid[:nrows, :ncols]
    >>> cnt_row, cnt_col = nrows / 2, ncols / 2
    >>> outer_disk_mask = ((row - cnt_row)**2 + (col - cnt_col)**2 >
    ...                    (nrows / 2)**2)
    >>> camera[outer_disk_mask] = 0

.. image:: ../auto_examples/numpy_operations/images/sphx_glr_plot_camera_numpy_001.png
    :width: 45%
    :target: ../auto_examples/numpy_operations/plot_camera_numpy.html

Boolean arithmetic can be used to define more complex masks: ::

    >>> lower_half = row > cnt_row
    >>> lower_half_disk = np.logical_and(lower_half, outer_disk_mask)
    >>> camera = data.camera()
    >>> camera[lower_half_disk] = 0 


Color images
------------

All of the above is true of color images, too: a color image is a
NumPy array, with an additional trailing dimension for the channels:

    >>> cat = data.chelsea()
    >>> type(cat)
    <type 'numpy.ndarray'>
    >>> cat.shape
    (300, 451, 3)

This shows that ``cat`` is a 300-by-451 pixel image with three
channels (red, green, and blue).
As before, we can get and set pixel values:

    >>> cat[10, 20]
    array([151, 129, 115], dtype=uint8)
    >>> # set the pixel at row 50, column 60 to black
    >>> cat[50, 60] = 0
    >>> # set the pixel at row 50, column 61 to green
    >>> cat[50, 61] = [0, 255, 0] # [red, green, blue]

We can also use 2D boolean masks for a 2D color image, as we did with
the grayscale image above:

.. plot::

    Using a 2D mask on a 2D color image

    >>> from skimage import data
    >>> cat = data.chelsea()
    >>> reddish = cat[:, :, 0] > 160
    >>> cat[reddish] = [0, 255, 0]
    >>> plt.imshow(cat)


.. _numpy-images-coordinate-conventions:

Coordinate conventions
----------------------

Because we represent images with numpy arrays, our coordinates must
match accordingly. Two-dimensional (2D) grayscale images (such as
`camera` above) are indexed by row and columns (abbreviated to either
``(row, col)`` or ``(r, c)``), with the lowest element ``(0, 0)`` at
the top-left corner. In various parts of the library, you will
also see ``rr`` and ``cc`` refer to lists of row and column
coordinates. We distinguish this from ``(x, y)``, which commonly denote
standard Cartesian coordinates, where ``x`` is the horizontal coordinate,
``y`` the vertical, and the origin is on the bottom left.
(Matplotlib, for example, uses this convention.)

In the case of color (or multichannel) images, the last dimension
contains the color information and is denoted ``channel`` or ``ch``.

Finally, for 3D images, such as videos, magnetic resonance imaging
(MRI) scans, or confocal microscopy, we refer to the leading dimension
as ``plane``, abbreviated as ``pln`` or ``p``.

These conventions are summarized below:

.. table:: Dimension name and order conventions in scikit-image

  =========================   ========================================
  Image type                  coordinates
  =========================   ========================================
  2D grayscale                (row, col)
  2D multichannel (eg. RGB)   (row, col, ch)
  3D grayscale                (pln, row, col)
  3D multichannel             (pln, row, col, ch)
  =========================   ========================================


Many functions in scikit-image operate on 3D images directly:

    >>> im3d = np.random.rand(100, 1000, 1000)
    >>> from skimage import morphology
    >>> from scipy import ndimage as ndi
    >>> seeds = ndi.label(im3d < 0.1)[0]
    >>> ws = morphology.watershed(im3d, seeds)

In many cases,
the third imaging dimension has lower resolution than the other two.
Some scikit-image functions provide a ``spacing`` keyword argument
to process these images:

    >>> from skimage import segmentation
    >>> slics = segmentation.slic(im3d, spacing=[5, 1, 1], multichannel=False)


Other times, processing must be done plane-wise. When planes are the
leading dimension, we can use the following syntax:

    >>> from skimage import filters
    >>> edges = np.zeros_like(im3d)
    >>> for pln, image in enumerate(im3d):
    ...     # iterate over the leading dimension (planes)
    ...     edges[pln] = filters.sobel(image)


Notes on array order
--------------------

Although the labeling of the axes seems arbitrary, it can have a
significant effect on speed of operations. This is because modern
processors never retrieve just one item from memory, but rather a
whole chunk of adjacent items. (This is called prefetching.)
Therefore, processing elements that are
next to each other in memory is faster than processing them
in a different order, even if the number of operations is the same:

    >>> def in_order_multiply(arr, scalar):
    ...     for plane in list(range(arr.shape[0])):
    ...         arr[plane, :, :] *= scalar
    ... 
    >>> def out_of_order_multiply(arr, scalar):
    ...     for plane in list(range(arr.shape[2])):
    ...         arr[:, :, plane] *= scalar
    ... 
    >>> import time
    >>> im3d = np.random.rand(100, 1024, 1024)
    >>> t0 = time.time(); x = in_order_multiply(im3d, 5); t1 = time.time()
    >>> print("%.2f seconds" % (t1 - t0))  # doctest: +SKIP
    0.14 seconds
    >>> im3d_t = np.transpose(im3d).copy() # place "planes" dimension at end
    >>> im3d_t.shape
    (1024, 1024, 100)
    >>> s0 = time.time(); x = out_of_order_multiply(im3d, 5); s1 = time.time()
    >>> print("%.2f seconds" % (s1 - s0))  # doctest: +SKIP
    1.18 seconds
    >>> print("Speedup: %.1fx" % ((s1 - s0) / (t1 - t0)))  # doctest: +SKIP
    Speedup: 8.6x


When the dimension you are iterating over is even larger, the
speedup is even more dramatic. It is worth thinking about this
*data locality* when writing algorithms. In particular, know that
scikit-image uses C-contiguous arrays unless otherwise specified, so
one should iterate along the last/rightmost dimension in the
innermost loop of the computation.

A note on time
--------------

Although scikit-image does not currently (0.11) provide functions to
work specifically with time-varying 3D data, our compatibility with
numpy arrays allows us to work quite naturally with a 5D array of the
shape (t, pln, row, col, ch):

    >>> for timepoint in image5d:  # doctest: +SKIP
    ...     # each timepoint is a 3D multichannel image
    ...     do_something_with(timepoint)


We can then supplement the above table as follows:

.. table:: Addendum to dimension names and orders in scikit-image

  ========================   ========================================
  Image type                 coordinates
  ========================   ========================================
  2D color video             (t, row, col, ch)
  3D multichannel video      (t, pln, row, col, ch)
  ========================   ========================================
