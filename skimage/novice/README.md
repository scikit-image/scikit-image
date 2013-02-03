skimage.novice
==============
A special Python image submodule for beginners.

Description
-----------
skimage.novice is a simple wrapper around the Python Image Library (PIL) for beginners.
It allows for easy loading, manipulating, and saving of image files.

**NOTE: This module uses the Cartesian coordinate system!**

Example
-------

    >>> from skimage.novice import picture  # special submodule for beginners

    >>> pic = novice.open('sample.png')     # create a picture object from a file
    >>> print pic.format                    # pictures know their format...
    'PNG'
    >>> print pic.path                      # ...and where they came from...
    '/Users/example/sample.png'
    >>> print pic.size                      # ...and their size
    (665, 500)
    >>> print pic.width                     # 'width' and 'height' also exposed
    665
    >>> pic.size = (200, 250)               # changing size automatically resizes
    >>> for pixel in pic:                   # can iterate over pixels
    >>> ... if ((pixel.red > 128) and           # pixels have RGB (values are 0-255)...
    >>> ...     (pixel.x < pic.width)):     # ...and know where they are
    >>> ...     pixel.red /= 2                  # pixel is an alias into the picture
    >>> ...
    >>> print pic.modified                  # pictures know if their pixels are dirty
    True
    >>> print pic.path                      # picture no longer corresponds to file
    None
    >>> pic[0:20, 0:20] = (0, 0, 0)         # overwrite lower-left rectangle with black
    >>> pic.save('sample-bluegreen.jpg')    # guess file type from suffix
    >>> print pic.path                      # picture now corresponds to file
    '/Users/example/sample-bluegreen.jpg'
    >>> print pic.format                    # ...has a different format
    JPEG
    >>> print pic.modified                  # and is now in sync
    False
