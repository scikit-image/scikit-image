To use this to build your Cython file use the commandline options:

.. sourcecode:: text

    $ python setup.py build_ext --inplace


**To do**

* add simple examples, adapt documentation on existing examples

* add/check existing doc

* adapting tests for each type of filter

**General remarks**

Basically these filters compute local histogram for each pixel. Histogram is build using a moving window in
order to limit redundant computation. The path followed by the moving window is given hereunder

 ...-----------------------\
/--------------------------/
\-------------------------- ...

A comparison is proposed with cmorph.dilate algorithm to show how computation costs evolve with respect to image size or
structuring element size. This implementation gives better results for large structuring elements.

A local histogram is update at each pixel by introducing pixel entering the structuring element border and
by removing those leaving it. The histogram size is 8bit (256 bins) for 8 bit images and 2 to 12 bit (up to 4096 bins)
for 16bit image depending on the image maximum value. Image with pixels higher than 4095 raise a ValueError.

The filter is applied up to the image border, the neighboorhood used is adjusted accordingly. The user may provide
a mask image (same size as input image) where non zero value are the part of the image participating the the
histogram computation. By default all the image is filtered.

