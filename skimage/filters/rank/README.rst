To do
-----

* add simple examples, adapt documentation on existing examples
* add/check existing doc
* adapting tests for each type of filter

General remarks
---------------

Basically these filters compute local histogram for each pixel. A histogram is
built using a moving window in order to limit redundant computation. The path
followed by the moving window is given hereunder

 ...-----------------------\
/--------------------------/
\-------------------------- ...

We compare grey.dilate to this histogram based method to show how
computational costs increase with respect to image size or structuring element
size. This implementation gives better results for large structuring elements.

The local histogram is updated at each pixel as the structuring element window
moves by, i.e. only those pixels entering and leaving the structuring element
update the local histogram. The histogram size is 8-bit (256 bins) for 8-bit
images and 2 to 16-bit for 16-bit images depending on the maximum value of the
image.

The filter is applied up to the image border, the neighborhood used is
adjusted accordingly. The user may provide a mask image (same size as input
image) where non zero values are the part of the image participating in the
histogram computation. By default the entire image is filtered.
