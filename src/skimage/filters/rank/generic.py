"""


General Description
-------------------

These filters compute the local histogram at each pixel, using a sliding window
similar to the method described in [1]_. A histogram is built using a moving
window in order to limit redundant computation. The moving window follows a
snake-like path:

...------------------------↘
↙--------------------------↙
↘--------------------------...

The local histogram is updated at each pixel as the footprint window
moves by, i.e. only those pixels entering and leaving the footprint
update the local histogram. The histogram size is 8-bit (256 bins) for 8-bit
images and 2- to 16-bit for 16-bit images depending on the maximum value of the
image.

The filter is applied up to the image border, the neighborhood used is
adjusted accordingly. The user may provide a mask image (same size as input
image) where non zero values are the part of the image participating in the
histogram computation. By default the entire image is filtered.

This implementation outperforms :func:`skimage.morphology.dilation`
for large footprints.

Input images will be cast in unsigned 8-bit integer or unsigned 16-bit integer
if necessary. The number of histogram bins is then determined from the maximum
value present in the image. Eventually, the output image is cast in the input
dtype, or the `output_dtype` if set.

To do
-----

* add simple examples, adapt documentation on existing examples
* add/check existing doc
* adapting tests for each type of filter


References
----------

.. [1] Huang, T. ,Yang, G. ;  Tang, G.. "A fast two-dimensional
       median filtering algorithm", IEEE Transactions on Acoustics, Speech and
       Signal Processing, Feb 1979. Volume: 27 , Issue: 1, Page(s): 13 - 18.


"""

from _skimage2.filters.rank.generic import (
    autolevel as autolevel,
    equalize as equalize,
    gradient as gradient,
    maximum as maximum,
    mean as mean,
    geometric_mean as geometric_mean,
    subtract_mean as subtract_mean,
    median as median,
    minimum as minimum,
    modal as modal,
    enhance_contrast as enhance_contrast,
    pop as pop,
    threshold as threshold,
    noise_filter as noise_filter,
    entropy as entropy,
    otsu as otsu,
)  # noqa: F401

__all__ = [
    'autolevel',
    'equalize',
    'gradient',
    'maximum',
    'mean',
    'geometric_mean',
    'subtract_mean',
    'median',
    'minimum',
    'modal',
    'enhance_contrast',
    'pop',
    'threshold',
    'noise_filter',
    'entropy',
    'otsu',
]

from _skimage2.filters.rank.generic import majority, sum, windowed_histogram  # noqa: F401

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
