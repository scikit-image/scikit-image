.. _data_types:

===================================
Image data types and what they mean
===================================

In ``skimage``, images are simply numpy_ arrays, which support a variety of
data types [1]_, *i.e.* "dtypes". To avoid distorting image intensities (see
`Rescaling intensity values`_), we assume that images use the following dtype
ranges:

=========  =================================
Data type  Range
=========  =================================
uint8      0 to 255
uint16     0 to 65535
uint32     0 to 2\ :sup:`32`
float      -1 to 1 or 0 to 1
int8       -128 to 127
int16      -32768 to 32767
int32      -2\ :sup:`31` to 2\ :sup:`31` - 1
=========  =================================

Note that float images should be restricted to the range -1 to 1 even though
the data type itself can exceed this range; all integer dtypes, on the other
hand, have pixel intensities that can span the entire data type range. With a
few exceptions, *64-bit (u)int images are not supported*.

Functions in ``skimage`` are designed so that they accept any of these dtypes,
but, for efficiency, *may return an image of a different dtype* (see `Output
types`_). If you need a particular dtype, ``skimage`` provides utility
functions that convert dtypes and properly rescale image intensities (see
`Input types`_). You should **never use** ``astype`` on an image, because it
violates these assumptions about the dtype range::

   >>> from skimage import img_as_float
   >>> image = np.arange(0, 50, 10, dtype=np.uint8)
   >>> print(image.astype(np.float)) # These float values are out of range.
   [  0.  10.  20.  30.  40.]
   >>> print(img_as_float(image))
   [ 0.          0.03921569  0.07843137  0.11764706  0.15686275]


Input types
===========

Although we aim to preserve the data range and type of input images, functions
may support only a subset of these data-types. In such
a case, the input will be converted to the required type (if possible), and
a warning message printed to the log if a memory copy is needed. Type
requirements should be noted in the docstrings.

The following utility functions in the main package are available to developers
and users:

=============  =================================
Function name  Description
=============  =================================
img_as_float   Convert to 64-bit floating point.
img_as_ubyte   Convert to 8-bit uint.
img_as_uint    Convert to 16-bit uint.
img_as_int     Convert to 16-bit int.
=============  =================================

These functions convert images to the desired dtype and *properly rescale their
values*. If conversion reduces the precision of the image, then a warning is
issued::

   >>> from skimage import img_as_ubyte
   >>> image = np.array([0, 0.5, 1], dtype=float)
   >>> img_as_ubyte(image)
   WARNING:dtype_converter:Possible precision loss when converting from
   float64 to uint8
   array([  0, 128, 255], dtype=uint8)


Additionally, some functions take a ``preserve_range`` argument where a range
conversion is convenient but not necessary. For example, interpolation in
``transform.warp`` requires an image of type float, which should have a range
in [0, 1]. So, by default, input images will be rescaled to this range.
However, in some cases, the image values represent physical measurements, such
as temperature or rainfall values, that the user does not want rescaled.
With ``preserve_range=True``, the original range of the data will be
preserved, even though the output is a float image. Users must then ensure
this non-standard image is properly processed by downstream functions, which
may expect an image in [0, 1].

    >>> from skimage import data
    >>> from skimage.transform import rescale
    >>> image = data.coins()
    >>> image.dtype, image.min(), image.max(), image.shape
    (dtype('uint8'), 1, 252, (303, 384))
    >>> rescaled = rescale(image, 0.5)
    >>> (rescaled.dtype, np.round(rescaled.min(), 4),
    ...  np.round(rescaled.max(), 4), rescaled.shape)
    (dtype('float64'), 0.0147, 0.9456, (152, 192))
    >>> rescaled = rescale(image, 0.5, preserve_range=True)
    >>> (rescaled.dtype, np.round(rescaled.min()),
    ...  np.round(rescaled.max()), rescaled.shape
    (dtype('float64'), 4.0, 241.0, (152, 192))


Output types
============

The output type of a function is determined by the function author and is
documented for the benefit of the user.  While this requires the user to
explicitly convert the output to whichever format is needed, it ensures that no
unnecessary data copies take place.

A user that requires a specific type of output (e.g., for display purposes),
may write::

   >>> from skimage import img_as_uint
   >>> out = img_as_uint(sobel(image))
   >>> plt.imshow(out)


Image processing pipeline
=========================

This dtype behavior allows you to string together any ``skimage`` function
without worrying about the image dtype.  On the other hand, if you want to use
a custom function that requires a particular dtype, you should call one of the
dtype conversion functions (here, ``func1`` and ``func2`` are ``skimage``
functions)::

   >>> from skimage import img_as_float
   >>> image = img_as_float(func1(func2(image)))
   >>> processed_image = custom_func(image)

Better yet, you can convert the image internally and use a simplified
processing pipeline::

   >>> def custom_func(image):
   ...     image = img_as_float(image)
   ...     # do something
   ...
   >>> processed_image = custom_func(func1(func2(image)))


Rescaling intensity values
==========================

When possible, functions should avoid blindly stretching image intensities
(e.g. rescaling a float image so that the min and max intensities are
0 and 1), since this can heavily distort an image. For example, if you're
looking for bright markers in dark images, there may be an image where no
markers are present; stretching its input intensity to span the full range
would make background noise look like markers.

Sometimes, however, you have images that should span the entire intensity
range but do not. For example, some cameras store images with 10-, 12-, or
14-bit depth per pixel. If these images are stored in an array with dtype
uint16, then the image won't extend over the full intensity range, and thus,
would appear dimmer than it should. To correct for this, you can use the
``rescale_intensity`` function to rescale the image so that it uses the full
dtype range::

   >>> from skimage import exposure
   >>> image = exposure.rescale_intensity(img10bit, in_range=(0, 2**10 - 1))

Here, the ``in_range`` argument is set to the maximum range for a 10-bit image.
By default, ``rescale_intensity`` stretches the values of ``in_range`` to match
the range of the dtype. ``rescale_intensity`` also accepts strings as inputs
to ``in_range`` and ``out_range``, so the example above could also be written
as::

   >>> image = exposure.rescale_intensity(img10bit, in_range='uint10')


Note about negative values
==========================

People very often represent images in signed dtypes, even though they only
manipulate the positive values of the image (e.g., using only 0-127 in an int8
image). For this reason, conversion functions *only spread the positive values*
of a signed dtype over the entire range of an unsigned dtype. In other words,
negative values are clipped to 0 when converting from signed to unsigned
dtypes. (Negative values are preserved when converting between signed dtypes.)
To prevent this clipping behavior, you should rescale your image beforehand::

   >>> image = exposure.rescale_intensity(img_int32, out_range=(0, 2**31 - 1))
   >>> img_uint8 = img_as_ubyte(image)

This behavior is symmetric: The values in an unsigned dtype are spread over
just the positive range of a signed dtype.


References
==========

.. _numpy: http://docs.scipy.org/doc/numpy/user/
.. [1] http://docs.scipy.org/doc/numpy/user/basics.types.html

