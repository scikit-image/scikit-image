Getting started
---------------

``scikit-image`` is an image processing Python package that works with
:mod:`numpy` arrays. The package is imported as ``skimage``: ::

    >>> import skimage

Most functions of ``skimage`` are found within submodules: ::

    >>> from skimage import data
    >>> camera = data.camera()

A list of submodules and functions is found on the `API reference
<http://scikit-image.org/docs/stable/api/api.html>`_ webpage.

Within scikit-image, images are represented as NumPy arrays, for
example 2-D arrays for grayscale 2-D images ::

    >>> type(camera)
    <type 'numpy.ndarray'>
    >>> # An image with 512 rows and 512 columns
    >>> camera.shape
    (512, 512)

The :mod:`skimage.data` submodule provides a set of functions returning
example images, that can be used to get started quickly on using
scikit-image's functions: ::

    >>> coins = data.coins()
    >>> from skimage import filters
    >>> threshold_value = filters.threshold_otsu(coins)
    >>> threshold_value
    107

Of course, it is also possible to load your own images as NumPy arrays
from image files, using :func:`skimage.io.imread`: ::

    >>> import os
    >>> filename = os.path.join(skimage.data_dir, 'moon.png')
    >>> from skimage import io
    >>> moon = io.imread(filename)

