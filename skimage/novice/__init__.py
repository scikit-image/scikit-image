"""
skimage.novice
==============
A special Python image submodule for beginners.

Description
-----------
``skimage.novice`` provides a simple image manipulation interface for
beginners.  It allows for easy loading, manipulating, and saving of image
files.

This module is primarily intended for teaching and differs significantly from
the normal, array-oriented image functions used by scikit-image.

.. note::

    This module uses the Cartesian coordinate system, where the origin is at
    the lower-left corner instead of the upper-right and the order is x, y
    instead of row, column.


Example
-------

We can create a Picture object open opening an image file
>>> from skimage import novice
>>> from skimage import data
>>> picture = novice.open(data.data_dir + '/chelsea.png')

Pictures know their format
>>> print picture.format
png

... and where they came from
>>> print picture.path.endswith('chelsea.png')
True

... and their size
>>> print picture.size
(451, 300)
>>> print picture.width
451

Changing `size` resizes the picture.
>>> picture.size = (200, 250)

You can iterate over pixels, which have RGB values between 0 and 255,
and know their location in the picture.
>>> for pixel in picture:
...     if (pixel.red > 128) and (pixel.x < picture.width):
...         pixel.red /= 2

Pictures know if they've been modified from the original file
>>> print picture.modified
True
>>> print picture.path
None

Pictures can be indexed like arrays
>>> picture[0:20, 0:20] = (0, 0, 0)

Saving the picture updates the path attribute, format, and modified state.
>>> picture.save('sample-bluegreen.jpg')
>>> print picture.path.endswith('sample-bluegreen.jpg')
True
>>> print picture.format
jpeg
>>> print picture.modified
False

"""
from ._novice import Picture, open, colors, color_dict


__all__ = ['Picture', 'open', 'colors', 'color_dict']
