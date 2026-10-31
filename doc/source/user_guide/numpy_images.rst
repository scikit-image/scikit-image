.. _numpy_images:

==================================
A crash course on NumPy for images
==================================

Images in ``scikit-image`` are represented by NumPy ndarrays. Hence, many
common operations can be achieved using standard NumPy methods for
manipulating arrays::

    >>> import skimage as ski
    >>> camera = ski.data.camera()
    >>> type(camera)
    <class 'numpy.ndarray'>

.. note::

    Labeled array-like data types, such as
    `pandas.DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_
    or `xarray.DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_,
    are not natively supported in ``scikit-image``. However, data stored in these types
    can be converted to ``numpy.ndarray`` with certain assumptions
    (see ``pandas.DataFrame.to_numpy()`` and ``xarray.DataArray.data``). Particularly,
    these conversions ignore the sampling coordinates (``DataFrame.index``,
    ``DataFrame.columns``, or ``DataArray.coords``), which may result in
    misrepresented data, for instance, when the original data points are irregularly
    spaced.

.. note::

    `We do not have native support for NumPy masked arrays
    <https://numpy.org/doc/stable/reference/maskedarray.baseclass.html#numpy.ma.MaskedArray>`_.
    Please convert images to plain ``numpy.ndarray``s, and handle masks by
    using separate Numpy arrays. Some functions accept mask arguments, and
    take Numpy arrays that define masks, but in many cases you'll want to
    handle the masking yourself (for example, by :ref:`using masks
    <numpy-images-masking>` to overwrite pixel values).

Retrieving the geometry of the image and the number of pixels::

    >>> camera.shape
    (512, 512)
    >>> camera.size
    262144

Retrieving statistical information about image intensity values::

    >>> import numpy as np
    >>>
    >>> np.min(camera), np.max(camera)
    (np.uint8(0), np.uint8(255))
    >>> np.mean(camera)
    np.float64(129.0607...)

NumPy arrays representing images can be of different integer or float
numerical types. See :ref:`data_types` for more information about these
types and how ``scikit-image`` treats them.

2D and 3D images
----------------

When we refer to a 2D image, we mean an image with two spatial dimensions,
rows and columns, and that can be represented as a 2D visual image.

In what immediately follows, we'll assume you have a simple one-color-channel
(grayscale) 2D image.

Later in the document we will generalize to other images.  For example, when
we refer (later) to a 3D image, we mean an image that has three spatial
dimensions that we could describe as rows, columns and planes.  We also cover
2D and 3D images with an extra color dimension; in that case we implement 2D
color (multichannel) images as arrays with three dimensions, and 3D color
images as arrays with four dimensions.

But for now, let us return to a simple grayscale (not color) 2D image,
represented by a Numpy array with two dimensions.

NumPy indexing
--------------

NumPy indexing can be used both for looking at the pixel values and to
modify them::

    >>> # Get the value of the pixel at the 10th row and 20th column
    >>> camera[10, 20]
    np.uint8(200)
    >>> # Set to black the pixel at the 3rd row and 10th column
    >>> camera[3, 10] = 0

.. note::

    Indexing above is Numpy indexing, where the first dimension
    (``camera.shape[0]``) corresponds to the rows of the image (and rows of
    the array), while the second (``camera.shape[1]``) corresponds to columns.
    The origin (``camera[0, 0]``) is at the top-left corner. This matches
    standard Numpy usage, and matrix/linear algebra notation. We emphasise
    this here because Numpy / matrix convention is opposite to another
    standard convention you will find in image analysis — Cartesian (x, y)
    coordinates — where `x` specifies left to right (columns) and `y`
    specifies top to bottom (rows). See :ref:`coordinates` for details of our
    conventions, and discussion of coordinate systems.

Beyond individual pixels, it is possible to access/modify values of
whole sets of pixels using the different indexing capabilities of NumPy.

Slicing::

    >>> # Set the first ten lines to "black" (0)
    >>> camera[:10] = 0

.. _numpy-images-masking:

Masking (indexing with masks of booleans)::

    >>> mask = camera < 87
    >>> # Set to "white" (255) the pixels where mask is True
    >>> camera[mask] = 255

Fancy indexing (indexing with sets of indices)::

    >>> inds_r = np.arange(len(camera))
    >>> inds_c = 4 * inds_r % len(camera)
    >>> camera[inds_r, inds_c] = 0

Masks are very useful when you need to select a set of pixels on which
to perform the manipulations. The mask can be any boolean array
of the same shape as the image (or a shape broadcastable to the image shape).
This can be used to define a region of interest, for example, a disk::

    >>> nrows, ncols = camera.shape
    >>> row, col = np.ogrid[:nrows, :ncols]
    >>> cnt_row, cnt_col = nrows / 2, ncols / 2
    >>> outer_disk_mask = ((row - cnt_row)**2 + (col - cnt_col)**2 >
    ...                    (nrows / 2)**2)
    >>> camera[outer_disk_mask] = 0

.. image:: ../auto_examples/numpy_operations/images/sphx_glr_plot_camera_numpy_001.png
    :width: 45%
    :target: ../auto_examples/numpy_operations/plot_camera_numpy.html

Boolean operations from NumPy can be used to define even more complex masks::

    >>> lower_half = row > cnt_row
    >>> lower_half_disk = np.logical_and(lower_half, outer_disk_mask)
    >>> camera = ski.data.camera()
    >>> camera[lower_half_disk] = 0


Color images
------------

All of the above remains true for color images. A color image is a NumPy array
with an additional dimension for the color channels.  Let's consider the
typical case where you have a 2D color image represented by a Numpy array with
three dimensions. By 2D, we mean an image with two spatial dimensions.  In the
simplest case, the first two axes of the array correspond to the (spatial)
rows and columns of the image (respectively), and the last corresponds to the
three Red Green Blue (RGB) color channels::

    >>> cat = ski.data.chelsea()
    >>> type(cat)
    <class 'numpy.ndarray'>
    >>> cat.shape
    (300, 451, 3)

This is a 2D spatial image represented as an array with three dimensions.  We
can also call this a 2D multichannel image, with the channel axis last.

The output above shows that ``cat`` is a 300-by-451 pixel image with three
color channels (RGB). As before, we can get and set the pixel values::

    >>> cat[10, 20]
    array([151, 129, 115], dtype=uint8)
    >>> # Set the pixel at (50th row, 60th column) to "black"
    >>> cat[50, 60] = 0
    >>> # set the pixel at (50th row, 61st column) to "green"
    >>> cat[50, 61] = [0, 255, 0]  # [red, green, blue]

We can also use 2D boolean masks for 2D multichannel images, as we did with
the grayscale image above:

.. plot::
   :caption: Using a 2D mask on a 2D color image

   import skimage as ski
   cat = ski.data.chelsea()
   reddish = cat[:, :, 0] > 160
   cat[reddish] = [0, 255, 0]
   plt.imshow(cat)

The example color images here, and all those included in :mod:`skimage.data`,
have channels stored along the last axis. Bear in mind that other software may
follow different conventions as to the default position of the channel axis.
The scikit-image library functions that support color images have a
``channel_axis`` argument that can be used to specify which axis of an array
(if any) corresponds to channels.  See :ref:`coordinates` for more on
specifying the channel axis.

Working with 3D images
----------------------

By 3D images, we mean images with three spatial dimensions.

Many functions in ``scikit-image`` can operate on 3D images directly.  Here we
segment the nuclei in a stack of confocal microscopy images.  ``cells3d``
returns an array of shape ``(plane, channel, row, column)``; channel 1 holds
the nuclei::

    >>> import numpy as np
    >>> import scipy as sp
    >>> import skimage as ski
    >>> nuclei = ski.data.cells3d()[:, 1]
    >>> nuclei.shape
    (60, 256, 256)

We next smooth the image to emphasize the spatially distributed nuclei
compared to the voxel-to-voxel noise::

    >>> smoothed = ski.filters.gaussian(nuclei, sigma=2)
    >>> seeds = sp.ndimage.label(smoothed > ski.filters.threshold_otsu(smoothed))[0]
    >>> ws = ski.segmentation.watershed(nuclei, seeds)
    >>> int(ws.max())  # One region per nucleus.
    23

In many cases, the third spatial dimension has lower resolution than the other
two, and this has consequences for the algorithm implemented by the function.
Some ``scikit-image`` functions provide a ``spacing`` keyword argument to help
handle this kind of data.  The microscope that recorded ``cells3d`` reports a
voxel size of 0.29 micrometers between planes, against 0.26 micrometers within
a plane.  In the following call, we use `spacing` to pass that geometry to
`slic`, and `channel_axis=None` to tell `slic` that none of the `nuclei` axes
correspond to color channels::

    >>> slics = ski.segmentation.slic(nuclei, n_segments=100,
    ...                               spacing=[0.29, 0.26, 0.26],
    ...                               channel_axis=None)

Other times, the processing must be done plane-wise. When planes are stacked
along the leading dimension, the following syntax can be used::

    >>> edges = np.empty(nuclei.shape, dtype=float)
    >>> for pln, image in enumerate(nuclei):
    ...     # Iterate over the leading dimension
    ...     edges[pln] = ski.filters.sobel(image)


Notes on the memory order of array dimensions
--------------------------------------------

Although the labeling of the axes might seem arbitrary, it can have a
significant effect on the speed of operations. This is because modern
processors implementing memory reads, do not retrieve just one item from
memory, but rather they retrieve a whole chunk of items that are contiguous in
memory (an operation called prefetching). Therefore, processing of elements
that are next to each other in memory may be much faster than processing them
when they are scattered in memory, even if the number of operations is the
same::

    >>> def in_order_multiply(arr, scalar):
    ...     for plane in list(range(arr.shape[0])):
    ...         arr[plane, :, :] *= scalar
    ...
    >>> def out_of_order_multiply(arr, scalar):
    ...     for plane in list(range(arr.shape[2])):
    ...         arr[:, :, plane] *= scalar
    ...
    >>> import time
    >>> rng = np.random.default_rng()
    >>> im3d = rng.random((100, 1024, 1024))
    >>> t0 = time.time(); x = in_order_multiply(im3d, 5); t1 = time.time()
    >>> print("%.2f seconds" % (t1 - t0))  # doctest: +SKIP
    0.14 seconds
    >>> s0 = time.time(); x = out_of_order_multiply(im3d, 5); s1 = time.time()
    >>> print("%.2f seconds" % (s1 - s0))  # doctest: +SKIP
    1.18 seconds
    >>> print("Speedup: %.1fx" % ((s1 - s0) / (t1 - t0)))  # doctest: +SKIP
    Speedup: 8.6x


When the last/rightmost dimension becomes even larger the speedup is even more
dramatic. It is worth thinking about this issue — known as *data locality* —
when developing algorithms. In particular, ``scikit-image`` uses C-contiguous
arrays by default. When using nested loops, the last/rightmost dimension of
a C-contiguous array has elements that are contiguous in memory, and you might
consider making sure they are in the innermost loop of the computation. In the
example above, the ``*=`` numpy operator iterates over all remaining
dimensions.
