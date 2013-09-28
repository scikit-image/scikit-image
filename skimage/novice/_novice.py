import os
import imghdr
from collections import namedtuple

import numpy as np
from skimage import io
from skimage import img_as_ubyte
from skimage.transform import resize
from skimage.color import color_dict
from skimage._shared import six


# Convert colors from `skimage.color` to uint8 and allow access through
# dict or a named tuple.
color_dict = dict((name, tuple(int(255 * c + 0.5) for c in rgb))
                  for name, rgb in color_dict.iteritems())
colors = namedtuple('colors', color_dict.keys())(**color_dict)


def open(path):
    """Return Picture object from the given image path."""
    return Picture(path=os.path.abspath(path))


def _verify_picture_index(index):
    """Raise error if picture index is not a 2D index/slice."""
    if not (isinstance(index, tuple) and len(index) == 2):
        raise IndexError("Expected 2D index but got {!r}".format(index))

    if all(isinstance(i, int) for i in index):
        return index

    # In case we need to fix the array index, convert tuple to list.
    index = list(index)

    for i, dim_slice in enumerate(index):
        # If either index is a slice, ensure index object returns 2D array.
        if isinstance(dim_slice, int):
            index[i] = dim_slice = slice(dim_slice, dim_slice + 1)

        if dim_slice.start is not None and dim_slice.start < 0:
            raise IndexError("Negative slicing not supported")

        if dim_slice.stop is not None and dim_slice.stop < 0:
            raise IndexError("Negative slicing not supported")

        if dim_slice.step is not None and dim_slice.step != 1:
            raise IndexError("Only a step size of 1 is supported")

    return tuple(index)


def array_to_xy_origin(image):
    """Return view of image transformed from array to Cartesian origin."""
    return np.transpose(image[::-1], (1, 0, 2))


def xy_to_array_origin(image):
    """Return view of image transformed from Cartesian to array origin."""
    return np.transpose(image[:, ::-1], (1, 0, 2))


class Pixel(object):
    """A single pixel in a Picture.

    Attributes
    ----------
    pic : Picture
        The Picture object that this pixel references.
    array : array_like
        Byte array with raw image data (RGB).
    x : int
        Horizontal coordinate of this pixel (left = 0).
    y : int
        Vertical coordinate of this pixel (bottom = 0).
    rgb : tuple
        RGB tuple with red, green, and blue components (0-255)

    """
    def __init__(self, pic, array, x, y, rgb):
        self._picture = pic
        self._x = x
        self._y = y
        self._red = self._validate(rgb[0])
        self._green = self._validate(rgb[1])
        self._blue = self._validate(rgb[2])

    @property
    def x(self):
        """Horizontal location of this pixel in the parent image(left = 0)."""
        return self._x

    @property
    def y(self):
        """Vertical location of this pixel in the parent image (bottom = 0)."""
        return self._y

    @property
    def red(self):
        """The red component of the pixel (0-255)."""
        return self._red

    @red.setter
    def red(self, value):
        self._red = self._validate(value)
        self._setpixel()

    @property
    def green(self):
        """The green component of the pixel (0-255)."""
        return self._green

    @green.setter
    def green(self, value):
        self._green = self._validate(value)
        self._setpixel()

    @property
    def blue(self):
        """The blue component of the pixel (0-255)."""
        return self._blue

    @blue.setter
    def blue(self, value):
        self._blue = self._validate(value)
        self._setpixel()

    @property
    def rgb(self):
        """The RGB color components of the pixel (3 values 0-255)."""
        return (self.red, self.green, self.blue)

    @rgb.setter
    def rgb(self, value):
        self._red, self._green, self._blue = (self._validate(v) for v in value)
        self._setpixel()

    def _validate(self, value):
        """Verifies that the pixel value is in [0, 255]."""
        try:
            value = int(value)
            if (value < 0) or (value > 255):
                raise ValueError()
        except ValueError:
            msg = "Expected an integer between 0 and 255, but got {0} instead!"
            raise ValueError(msg.format(value))

        return value

    def _setpixel(self):
        self._picture.xy_array[self._x, self._y] = self.rgb
        self._picture._array_modified()

    def __repr__(self):
        args = self.red, self.green, self.blue
        return "Pixel(red={0}, green={1}, blue={2})".format(*args)


class Picture(object):
    """A 2-D picture made up of pixels.

    Attributes
    ----------
    path : str
        Path to an image file to load.
    array : array
        Raw RGB image data [0-255], with origin at top-left.
    xy_array : array
        Raw RGB image data [0-255], with origin at bottom-left.

    Examples
    --------
    Load an image from a file
    >>> from skimage import novice
    >>> from skimage import data
    >>> picture = novice.open(data.data_dir + '/chelsea.png')

    Create a blank 100 pixel wide, 200 pixel tall white image
    >>> pic = Picture.from_size((100, 200), color=(255, 255, 255))

    Use numpy to make an RGB byte array (shape is height x width x 3)
    >>> import numpy as np
    >>> data = np.zeros(shape=(200, 100, 3), dtype=np.uint8)
    >>> data[:, :, 0] = 255  # Set red component to maximum
    >>> pic = Picture(array=data)

    Get the bottom-left pixel
    >>> pic[0, 0]
    Pixel(red=255, green=0, blue=0)

    Get the top row of the picture
    >>> pic[:, pic.height-1]
    Picture(100 x 1)

    Set the bottom-left pixel to black
    >>> pic[0, 0] = (0, 0, 0)

    Set the top row to red
    >>> pic[:, pic.height-1] = (255, 0, 0)

    """
    def __init__(self, path=None, array=None, xy_array=None):
        self._modified = False
        self.scale = 1
        self._path = None
        self._format = None

        n_args = len([a for a in [path, array, xy_array] if a is not None])
        if n_args != 1:
            msg = "Must provide a single keyword arg (path, array, xy_array)."
            ValueError(msg)
        elif path is not None:
            self.array = img_as_ubyte(io.imread(path))
            self._path = path
            self._format = imghdr.what(path)
        elif array is not None:
            self.array = array
        elif xy_array is not None:
            self.xy_array = xy_array

    @staticmethod
    def from_size(size, color='black'):
        """Return a Picture of the specified size and a uniform color.

        Parameters
        ----------
        size : tuple
            Width and height of the picture in pixels.
        color : tuple or str
            RGB tuple with the fill color for the picture [0-255] or a valid
            key in `color_dict`.
        """
        if isinstance(color, six.string_types):
            color = color_dict[color]
        rgb_size = tuple(size) + (3,)
        array = np.ones(rgb_size, dtype=np.uint8) * color
        return Picture(array=array)

    @property
    def array(self):
        """Image data stored as numpy array."""
        return self._array

    @array.setter
    def array(self, array):
        self._array = array
        self._xy_array = array_to_xy_origin(array)

    @property
    def xy_array(self):
        """Image data stored as numpy array with origin at the bottom-left."""
        return self._xy_array

    @xy_array.setter
    def xy_array(self, array):
        self._xy_array = array
        self._array = xy_to_array_origin(array)

    def save(self, path):
        """Saves the picture to the given path.

        Parameters
        ----------
        path : str
            Path (with file extension) where the picture is saved.
        """
        io.imsave(path, self._rescale(self.array))
        self._modified = False
        self._path = os.path.abspath(path)
        self._format = imghdr.what(path)

    @property
    def path(self):
        """The path to the picture."""
        return self._path

    @property
    def modified(self):
        """True if the picture has changed."""
        return self._modified

    def _array_modified(self):
        self._modified = True
        self._path = None

    @property
    def format(self):
        """The image format of the picture."""
        return self._format

    @property
    def size(self):
        """The size (width, height) of the picture."""
        return self.xy_array.shape[:2]

    @size.setter
    def size(self, value):
        # Don't resize if no change in size
        if (value[0] != self.width) or (value[1] != self.height):
            # skimage dimensions are flipped: y, x
            new_size = (int(value[1]), int(value[0]))
            new_array = resize(self.array, new_size, order=0)
            self.array = img_as_ubyte(new_array)

            self._array_modified()

    @property
    def width(self):
        """The width of the picture."""
        return self.size[0]

    @width.setter
    def width(self, value):
        self.size = (value, self.height)

    @property
    def height(self):
        """The height of the picture."""
        return self.size[1]

    @height.setter
    def height(self, value):
        self.size = (self.width, value)

    def _repr_png_(self):
        return io.Image(self._rescale(self.array))._repr_png_()

    def show(self):
        """Display the image."""
        io.imshow(self._rescale(self.array))
        io.show()

    def _makepixel(self, x, y):
        """Create a Pixel object for a given x, y location."""
        rgb = self.xy_array[x, y]
        return Pixel(self, self.array, x, y, rgb)

    def _rescale(self, array):
        """Rescale image according to scale factor."""
        if self.scale == 1:
            return array
        new_size = (self.height * self.scale, self.width * self.scale)
        return img_as_ubyte(resize(array, new_size, order=0))

    def _get_channel(self, channel):
        """Return a specific dimension out of the raw image data slice."""
        return self._array[:, :, channel]

    def _set_channel(self, channel, value):
        """Set a specific dimension in the raw image data slice."""
        self._array[:, :, channel] = value

    @property
    def red(self):
        """The red component of the pixel (0-255)."""
        return self._get_channel(0).ravel()

    @red.setter
    def red(self, value):
        self._set_channel(0, value)

    @property
    def green(self):
        """The green component of the pixel (0-255)."""
        return self._get_channel(1).ravel()

    @green.setter
    def green(self, value):
        self._set_channel(1, value)

    @property
    def blue(self):
        """The blue component of the pixel (0-255)."""
        return self._get_channel(2).ravel()

    @blue.setter
    def blue(self, value):
        self._set_channel(2, value)

    @property
    def rgb(self):
        """The RGB color components of the pixel (3 values 0-255)."""
        return self.xy_array

    @rgb.setter
    def rgb(self, value):
        self.xy_array[:] = value

    def __iter__(self):
        """Iterates over all pixels in the image."""
        for x in xrange(self.width):
            for y in xrange(self.height):
                yield self._makepixel(x, y)

    def __getitem__(self, xy_index):
        """Return `Picture`s for slices and `Pixel`s for indexes."""
        xy_index = _verify_picture_index(xy_index)
        if all(isinstance(index, int) for index in xy_index):
            if any(index < 0 for index in xy_index):
                raise IndexError("Negative indices not supported")
            return self._makepixel(*xy_index)
        else:
            return Picture(xy_array=self.xy_array[xy_index])

    def __setitem__(self, xy_index, value):
        xy_index = _verify_picture_index(xy_index)
        if isinstance(value, tuple):
            self[xy_index].rgb = value
        elif isinstance(value, Picture):
            self.xy_array[xy_index] = value.xy_array
        else:
            raise TypeError("Invalid value type")
        self._array_modified()

    def __eq__(self, other):
        if not isinstance(other, Picture):
            raise NotImplementedError()
        return np.all(self.array == other.array)

    def __repr__(self):
        return "Picture({0} x {1})".format(*self.size)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
