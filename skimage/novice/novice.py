import os, numpy as np, imghdr
from skimage.util import img_as_ubyte
import skimage.io, skimage.transform

# ================================================== 

class Pixel(object):
    def __init__(self, pic, image, x, y, rgb):
        self._picture = pic
        self._image = image
        self._x = x
        self._y = y
        self._red = self._validate(rgb[0])
        self._green = self._validate(rgb[1])
        self._blue = self._validate(rgb[2])

    @property
    def x(self):
        """Gets the horizontal location (left = 0)"""
        return self._x

    @property
    def y(self):
        """Gets the vertical location (bottom = 0)"""
        return self._y

    @property
    def red(self):
        """Gets or sets the red component"""
        return self._red

    @red.setter
    def red(self, value):
        self._red = self._validate(value)
        self._setpixel()

    @property
    def green(self):
        """Gets or sets the green component"""
        return self._green

    @green.setter
    def green(self, value):
        self._green = self._validate(value)
        self._setpixel()

    @property
    def blue(self):
        """Gets or sets the blue component"""
        return self._blue

    @blue.setter
    def blue(self, value):
        self._blue = self._validate(value)
        self._setpixel()

    @property
    def rgb(self):
        return (self.red, self.green, self.blue)

    @rgb.setter
    def rgb(self, value):
        """Gets or sets the color with an (r, g, b) tuple"""
        for v in value:
            self._validate(v)

        self._red = value[0]
        self._green = value[1]
        self._blue = value[2]
        self._setpixel()

    def _validate(self, value):
        """Verifies that the pixel value is in [0, 255]"""
        try:
            value = int(value)
            if (value < 0) or (value > 255):
                raise ValueError()
        except ValueError:
            raise ValueError("Expected an integer between 0 and 255, but got {0} instead!".format(value))

        return value

    def _setpixel(self):
        """
        Sets the actual pixel value in the picture.
        NOTE: Using Cartesian coordinate system!
        """
        # skimage dimensions are flipped: y, x
        self._image[self._picture.height - self._y - 1, self._x] = \
                (self.red, self.green, self.blue)

        # Modified pictures lose their paths
        self._picture._picture_path = None
        self._picture._picture_modified = True

    def __repr__(self):
        return "Pixel (red: {0}, green: {1}, blue: {2})".format(self.red, self.green, self.blue)

# ================================================== 

class PixelGroup(object):
    def __init__(self, pic, key):
        self._pic = pic

        # Use a slice so that the _getdim and _setdim functions can index
        # consistently.
        if isinstance(key[0], int):
            key = (slice(key[0], key[0] + 1), key[1])

        if isinstance(key[1], int):
            key = (key[0], slice(key[1], key[1] + 1))

        for dim_slice in key:
            if dim_slice.start is not None and dim_slice.start < 0:
                raise IndexError("Negative slicing not supported")

            if dim_slice.stop is not None and dim_slice.stop < 0:
                raise IndexError("Negative slicing not supported")

            if dim_slice.step is not None and dim_slice.step != 1:
                raise IndexError("Only a step size of 1 is supported")

        # ===========
        # Flip y axis
        # ===========
        y_slice = key[1]
        start = y_slice.start if y_slice.start is not None else 0
        stop = y_slice.stop if y_slice.stop is not None else pic.height

        start = pic.height - start - 1
        stop = pic.height - stop

        key = (key[0], slice(stop, start + 1, y_slice.step))

        # skimage dimensions are flipped: y, x
        self._key = (key[1], key[0])
        self._image = pic._image

        # Save slice for _getdim operations.
        # This allows you to swap parts of an image.
        self._slice = self._image[self._key[0], self._key[1]]

        shape = self._getdim(0).shape
        self.size = (shape[1], shape[0])

    def _getdim(self, dim):
        return self._slice[:, :, dim]

    def _setdim(self, dim, value):
        self._image[self._key[0], self._key[1], dim] = value

    @property
    def red(self):
        """Gets or sets the red component"""
        return self._getdim(0).ravel()

    @red.setter
    def red(self, value):
        self._setdim(0, value)

    @property
    def green(self):
        """Gets or sets the green component"""
        return self._getdim(1).ravel()

    @green.setter
    def green(self, value):
        self._setdim(1, value)

    @property
    def blue(self):
        """Gets or sets the blue component"""
        return self._getdim(2).ravel()

    @blue.setter
    def blue(self, value):
        self._setdim(2, value)

    @property
    def rgb(self):
        return self._getdim(None)

    @rgb.setter
    def rgb(self, value):
        """Gets or sets the color with an (r, g, b) tuple"""
        self._setdim(None, value)

    def __iter__(self):
        x_idx = range(self._pic.width)[self._key[0]]
        y_idx = range(self._pic.height)[self._key[1]]

        for x in x_idx:
            for y in y_idx:
                yield self._pic._makepixel((x, y))

    def __repr__(self):
        return "PixelGroup ({0} pixels)".format(self.size[0] * self.size[1])

# ================================================== 

class Picture(object):
    def __init__(self, path=None, size=None, color=None, image=None):
        """
        If 'path' is provided, open that file (the normal case).
        If 'size' is provided instead, create an image of that size.
        If 'color' is provided as well as 'size', initialize the
        created image to that color; otherwise, initialize to black.
        If 'image' is provided, use it as the underlying image data.
        Cannot provide more than one of 'path' and 'size' and 'image'.
        Can only provide 'color' if 'size' provided.
        """

        # Can only provide either path or size, but not both.
        if (path and size) or (path and image) or (size and image):
            assert False, "Can only provide path, size, or image."

        # Opening a particular file.  Convert the image to RGB
        # automatically so (r, g, b) tuples can be used
        # everywhere.
        elif path is not None:
            self._image = skimage.img_as_ubyte(skimage.io.imread(path))
            self._path = path
            self._format = imghdr.what(path)

        # Creating a particular size of image.
        elif size is not None:
            if color is None:
                color = (0, 0, 0)

            # skimage dimensions are flipped: y, x
            self._image = np.zeros((size[1], size[0], 3), "uint8")
            self._image[:, :] = color
            self._path = None
            self._format = None
        elif image is not None:
            self._image = image
            self._path = None
            self._format = None

        # Must have provided 'path', 'size', or 'image'.
        else:
            assert False, "Must provide one of path, size, or image."

        # Common setup.
        self._modified = False
        self._inflation = 1

    @staticmethod
    def from_path(path):
        return Picture(path=path)

    @staticmethod
    def from_color(color, size):
        return Picture(color=color, size=size)

    @staticmethod
    def from_image(image):
        return Picture(image=image)

    def save(self, path):
        """Saves the picture to the given path."""
        skimage.io.imsave(path, self._inflate(self._image))
        self._modified = False
        self._path = os.path.abspath(path)
        self._format = imghdr.what(path)

    @property
    def path(self):
        """Gets the path of the picture"""
        return self._path

    @property
    def modified(self):
        """Gets a value indicating if the picture has changed"""
        return self._modified

    @property
    def format(self):
        """Gets the format of the picture (e.g., PNG)"""
        return self._format

    @property
    def size(self):
        """Gets or sets the size of the picture with a (width, height) tuple"""
        # skimage dimensions are flipped: y, x
        return (self._image.shape[1], self._image.shape[0])

    @size.setter
    def size(self, value):
        try:
            # Don't resize if no change in size
            if (value[0] != self.width) or (value[1] != self.height):
                # skimage dimensions are flipped: y, x
                self._image = skimage.img_as_ubyte(skimage.transform.resize(self._image,
                    (value[1], value[0]), order=0))

                self._modified = True
                self._path = None
        except TypeError:
            raise TypeError("Expected (width, height), but got {0} instead!".format(value))

    @property
    def width(self):
        """Gets or sets the width of the image"""
        return self.size[0]

    @width.setter
    def width(self, value):
        self.size = (value, self.height)

    @property
    def height(self):
        """Gets or sets the height of the image"""
        return self.size[1]

    @height.setter
    def height(self, value):
        self.size = (self.width, value)

    @property
    def inflation(self):
        """Gets or sets the inflation factor (each pixel will be an NxN block for factor N)"""
        return self._inflation

    @inflation.setter
    def inflation(self, value):
        try:
            value = int(value)
            if value < 0:
                raise ValueError()
            self._inflation = value
        except ValueError:
            raise ValueError("Expected inflation factor to be an integer greater than zero")

    def _repr_html_(self):
        """Returns an Image for display in an IPython notebook"""
        return skimage.io.Image(self._inflate(self._image))

    def _repr_png_(self):
        """Returns an Image for display in an IPython console"""
        return skimage.io.Image(self._inflate(self._image))

    def show(self):
        """Displays the image in a separate window"""
        skimage.io.imshow(self._inflate(self._image))

    def _makepixel(self, xy):
        """
        Creates a Pixel object for a given x, y location.
        NOTE: Using Cartesian coordinate system!
        """
        # skimage dimensions are flipped: y, x
        rgb = self._image[self.height - xy[1] - 1, xy[0]]
        return Pixel(self, self._image, xy[0], xy[1], rgb)

    def _inflate(self, img):
        """Returns resized image using inflation factor (nearest neighbor)"""
        if self._inflation == 1:
            return img

        # skimage dimensions are flipped: y, x
        return img_as_ubyte(skimage.transform.resize(self._image, 
            (self.height * self._inflation,
             self.width * self._inflation), order=0))

    def __iter__(self):
        """Iterates over all pixels in the image"""
        for x in xrange(self.width):
            for y in xrange(self.height):
                yield self._makepixel((x, y))

    def __getitem__(self, key):
        """
        Gets pixels using 2D int or slice notations.
        Examples:
            pic[0, 0]              # Bottom-left pixel
            pic[:, pic.height-1]   # Top row
            pic[::2, ::2]          # Every other pixel
        """
        if isinstance(key, tuple) and len(key) == 2:
            if isinstance(key[0], int) and isinstance(key[1], int):
                if key[0] < 0 or key[1] < 0:
                    raise IndexError("Negative indices not supported")

                return self._makepixel((key[0], key[1]))
            else:
                return PixelGroup(self, key)
        else:
            raise TypeError("Invalid key type")

    def __setitem__(self, key, value):
        """
        Sets pixel values using 2D int or slice notations.
        Examples:
            pic[0, 0] = (0, 0, 0)               # Make bottom-left pixel black
            pic[:, pic.height-1] = (255, 0, 0)  # Make top row red
            pic[::2, ::2] = (255, 255, 255)     # Make every other pixel white
        """
        if isinstance(key, tuple) and len(key) == 2:
            if isinstance(value, tuple):
                self[key[0], key[1]].rgb = value
            elif isinstance(value, PixelGroup):
                src_key = self[key[0], key[1]]._key
                self._image[src_key] = value._image[value._key]
            else:
                raise TypeError("Invalid value type")
        else:
            raise TypeError("Invalid key type")

    def __repr__(self):
        return "Picture (format: {0}, path: {1}, modified: {2})".format(self.format, self.path, self.modified)

