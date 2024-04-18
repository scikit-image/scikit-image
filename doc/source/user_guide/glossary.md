# Glossary

Work in progress

```{glossary}

array
    Numerical array, provided by the {obj}`numpy.ndarray` object. In
    ``scikit-image``, images are NumPy arrays with dimensions that
    correspond to spatial dimensions of the image, and color channels for
    color images. See {ref}`numpy_images`.

channel
    Typically used to refer to a single color channel in a color image. RGBA
    images have an additional alpha (transparency) channel. Functions use a
    ``channel_axis`` argument to specify which axis of an array corresponds
    to channels. Images without channels are indicated via
    ``channel_axis=None``. Aside from the functions in ``skimage.color``, most
    functions with a ``channel_axis`` argument just apply the same operation
    across each channel. In this case, the "channels" do not strictly need to
    represent color or alpha information, but may be any generic batch
    dimension over which to operate.

circle
    The perimeter of a {term}`disk`.

contour
    Curve along which a 2-D image has a constant value. The interior
    (resp. exterior) of the contour has values greater (resp. smaller)
    than the contour value.

contrast
    Differences of intensity or color in an image, which make objects
    distinguishable. Several functions to manipulate the contrast of an
    image are available in {mod}`skimage.exposure`. See {ref}`exposure`.

disk
    A filled-in {term}`circle`.

float
    Representation of real numbers, for example as {obj}`numpy.float32` or
    {obj}`numpy.float64`. See {ref}`data_types`. Some operations on images
    need a float datatype (such as multiplying image values with
    exponential prefactors in {func}`skimage.filters.gaussian`), so that
    images of integer type are often converted to float type internally. Also
    see {term}`int` values.

float values
    See {term}`float`.

histogram
    For an image, histogram of intensity values, where the range of
    intensity values is divided into bins and the histogram counts how
    many pixel values fall in each bin. See
    {func}`skimage.exposure.histogram`.

int
    Representation of integer numbers, which can be signed or not, and
    encoded on one, two, four or eight bytes according to the maximum value
    which needs to be represented. In ``scikit-image``, the most common
    integer types are {obj}`numpy.int64` (for large integer values) and
    {obj}`numpy.uint8` (for small integer values, typically images of labels
    with less than 255 labels). See {ref}`data_types`.

int values
    See {term}`int`.

iso-valued contour
    See {term}`contour`.

labels
    An image of labels is of integer type, where pixels with the same
    integer value belong to the same object. For example, the result of a
    segmentation is an image of labels. {func}`skimage.measure.label` labels
    connected components of a binary image and returns an image of
    labels. Labels are usually contiguous integers, and
    {func}`skimage.segmentation.relabel_sequential` can be used to relabel
    arbitrary labels to sequential (contiguous) ones.

label image
    See {term}`labels`.

pixel
    Smallest element of an image. An image is a grid of pixels, and the
    intensity of each pixel is variable. A pixel can have a single
    intensity value in grayscale images, or several channels for color
    images. In ``scikit-image``, pixels are the individual elements of
    ``numpy arrays`` (see {ref}`numpy_images`).
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
