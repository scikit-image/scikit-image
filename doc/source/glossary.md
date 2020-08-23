Glossary
========

Work in progress

```{glossary}

array 
    Numerical array, provided by the {class}`numpy.ndarray` object. In
    ``scikit-image``, images are NumPy arrays, which dimensions
    correspond to spatial dimensions of the image, and color channels for
    color images. See {ref}`numpy`. 

contour
iso-valued contour
    Curve along which a 2-D image has a constant value. The interior
    (resp. exterior) of the contour has values greater (resp. smaller)
    than the contour value. 

contrast
    Differences of intensity or color in an image, which make objects
    distinguishable. Several functions to manipulate the contrast of an
    image are available in {mod}`skimage.exposure`. See {ref}`exposure`.

float
float values
    Representation of real numbers, for example as {obj}`np.float32` or
    {obj}`np.float64`. See {ref}`data_types`. Some operations on images
    need a float datatype (such as multiplying image values with
    exponential prefactors in {func}`filters.gaussian`), so that 
    images of integer type are often converted to float type internally. Also
    see {term}`int` values.

histogram
    For an image, histogram of intensity values, where the range of
    intensity values is divided into bins and the histogram counts how
    many pixel values fall in each bin. See
    {func}`exposure.histogram`.

int
int values
    Representation of integer numbers, which can be signed or not, and
    encoded on one, two, four or eight bytes according to the maximum value
    which needs to be represented. In ``scikit-image``, the most common
    integer types are {obj}`np.int64` (for large integer values) and
    {obj}`np.uint8` (for small integer values, typically images of labels
    with less than 255 labels). See {ref}`data_types`. 

labels
label image
    An image of labels is of integer type, where pixels with the same
    integer value belong to the same object. For example, the result of a
    segmentation is an image of labels. {func}`measure.label` labels
    connected components of a binary image and returns an image of
    labels. Labels are usually contiguous integers, and
    {func}`segmentation.relabel_sequential` can be used to relabel
    arbitrary labels to sequential (contiguous) ones.

pixel
    Smallest element of an image. An image is a grid of pixels, and the
    intensity of each pixel is variable. A pixel can have a single
    intensity value in grayscale images, or several channels for color
    images. In ``scikit-image``, pixels are the individual elements of
    ``numpy arrays`` (see {ref}`numpy`).
    Also see {term}`voxel`.

segmentation
    Partitioning an image into multiple objects (segments), for
    example an object of interest and its background. The output of a
    segmentation is typically an image of {term}`labels`, where
    the pixels of different objects have been attributed different
    integer labels. Several segmentation algorithms are available in
    {mod}`skimage.segmentation`.

voxel
    {term}`pixel` (smallest element of an image) of a
    three-dimensional image.
```
