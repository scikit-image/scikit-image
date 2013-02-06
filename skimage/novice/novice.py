import os, skimage, numpy as np
import skimage.io, skimage.transform
from itertools import islice, product

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
        self._image[self._x, self._picture.height - self._y - 1] = \
                (self.red, self.green, self.blue)

        # Modified pictures lose their paths
        self._picture._picture_path = None
        self._picture._picture_modified = True

    def __repr__(self):
        return "Pixel (red: {0}, green: {1}, blue: {2})".format(self.red, self.green, self.blue)

# ================================================== 

class PixelGroup(object):
    def __init__(self, pixels):
        self._pixels = pixels

    @property
    def x(self):
        """Gets the horizontal locations (left = 0)"""
        return [p.x for p in self]

    @property
    def y(self):
        """Gets the vertical locations (bottom = 0)"""
        return [p.y for p in self]

    @property
    def red(self):
        """Gets or sets the red component"""
        return [p.red for p in self]

    @red.setter
    def red(self, value):
        for p in self:
            p.red = value

    @property
    def green(self):
        """Gets or sets the green component"""
        return [p.green for p in self]

    @green.setter
    def green(self, value):
        for p in self:
            p.green = value

    @property
    def blue(self):
        """Gets or sets the blue component"""
        return [p.blue for p in self]

    @blue.setter
    def blue(self, value):
        for p in self:
            p.blue = value

    @property
    def rgb(self):
        return [p.rgb for p in self]

    @rgb.setter
    def rgb(self, value):
        """Gets or sets the color with an (r, g, b) tuple"""
        for p in self:
            p.rgb = value

    def __iter__(self):
        return iter(self._pixels)

    def __repr__(self):
        return "PixelGroup ({0} pixels)".format(len(self._pixels))

# ================================================== 

class Picture(object):
    def __init__(self, path=None, size=None, color=None):
        """
        If 'path' is provided, open that file (the normal case).
        If 'size' is provided instead, create an image of that size.
        If 'color' is provided as well as 'size', initialize the
        created image to that color; otherwise, initialize to black.
        Cannot provide both 'path' and 'size'.
        Can only provide 'color' if 'size' provided.
        """

        # Can only provide either path or size, but not both.
        if path and size:
            assert False, "Can only provide either path or size, not both."

        # Opening a particular file.  Convert the image to RGB
        # automatically so (r, g, b) tuples can be used
        # everywhere.
        elif path is not None:
            self._image = skimage.img_as_ubyte(skimage.io.imread(path))
            self._path = path
            self._format = None

        # Creating a particular size of image.
        elif size is not None:
            if color is None:
                color = (0, 0, 0)
            self._image = np.zeros((size[0], size[1], 3), "uint8")
            self._image[:, :] = color
            self._path = None
            self._format = None

        # Must have provided either 'path' or 'size'.
        else:
            assert False, "Must provide one of path or size."

        # Common setup.
        self._modified = False
        self._inflation = 1

    def save(self, path):
        """Saves the picture to the given path."""
        skimage.io.imsave(path, self._inflate(self._image))
        self._modified = False
        self._path = os.path.abspath(path)

        # Need to re-open the image to get the format
        # for some reason (likely because we converted to RGB).
        #self._format = Image.open(path).format

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
        return self._image.shape[:2]

    @size.setter
    def size(self, value):
        try:
            # Don't resize if no change in size
            if (value[0] != self.width) or (value[1] != self.height):
                self._image = skimage.img_as_ubyte(skimage.transform.resize(self._image, value, order=0))
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

    def show(self):
        """Returns an IPython image of the picture for display in an IPython notebook"""
        return skimage.io.Image(self._image)

    def _makepixel(self, xy):
        """
        Creates a Pixel object for a given x, y location.
        NOTE: Using Cartesian coordinate system!
        """
        (x,y) = xy
        rgb = self._image[x, self.height - y - 1]
        return Pixel(self, self._image, x, y, rgb)

    def _inflate(self, img):
        """Returns resized image using inflation factor (nearest neighbor)"""
        return skimage.img_as_ubyte(skimage.transform.resize(self._image, 
                (self.width * self._inflation,
                 self.height * self._inflation), order=0))

    def _negidx(self, idx, bound):
        """Handles negative indices by wrapping around bound"""
        if (idx is None) or idx >= 0:
            return idx
        else:
            return bound + idx

    def __iter__(self):
        """Iterates over all pixels in the image"""
        for x in xrange(self.width):
            for y in xrange(self.height):
                yield self._makepixel((x, y))

    def _keys(self, key):
        """
        Takes a key for __getitem__ or __setitem__ and
        validates it.  If valid, returns either a pair of ints
        or an iterator of pairs of ints.
        """
        if isinstance(key, tuple) and len(key) == 2:
            slx = key[0]
            sly = key[1]

            if ((isinstance(slx, int) or isinstance(slx, slice)) and
                (isinstance(sly, int) or isinstance(sly, slice))):
                if isinstance(slx, int):
                    slx = self._negidx(slx, self.width)
                    if (slx < 0) or (slx >= self.width):
                        raise IndexError("Index out of range")

                if isinstance(sly, int):
                    sly = self._negidx(sly, self.height)
                    if (sly < 0) or (sly >= self.height):
                        raise IndexError("Index out of range")

                # self[x, y]
                if isinstance(slx, int) and isinstance(sly, int):
                    return (slx, sly)
                
                if isinstance(slx, int):
                    slx = [slx]
                else: # slice (allow negative indices)
                    start = self._negidx(slx.start, self.width)
                    stop  = self._negidx(slx.stop,  self.width)
                    slx = islice(xrange(self.width), start, stop, slx.step)
                    
                if isinstance(sly, int):
                    sly = [sly]
                else: # slice (allow negative indices)
                    start = self._negidx(sly.start, self.height)
                    stop  = self._negidx(sly.stop,  self.height)
                    sly = islice(xrange(self.height), start, stop, sly.step)

                return product(slx, sly)

        # if either left or right is not an int or a slice, or
        # if the key is not a pair, fall through
        raise TypeError("Invalid key type")

    def __getitem__(self, key):
        """
        Gets pixels using 2D int or slice notations.
        Examples:
            pic[0, 0]      # Bottom-left pixel
            pic[:, -1]     # Top row
            pic[::2, ::2]  # Every other pixel
        """
        keys = self._keys(key)
        if isinstance(keys, tuple):
            return self._makepixel(keys)
        else:
            return PixelGroup(map(self._makepixel, keys))

    def __setitem__(self, key, value):
        """
        Sets pixel values using 2D int or slice notations.
        Examples:
            pic[0, 0] = (0, 0, 0)            # Make bottom-left pixel black
            pic[:, -1] = (255, 0, 0)         # Make top row red
            pic[::2, ::2] = (255, 255, 255)  # Make every other pixel white
        """
        keys = self._keys(key)
        if isinstance(keys, tuple):
            pixel = self[keys[0], keys[1]]
            pixel.rgb = value
        else:
            for (x,y) in keys:
                pixel = self[x,y]
                pixel.rgb = value

    def __repr__(self):
        return "Picture (format: {0}, path: {1}, modified: {2})".format(self.format, self.path, self.modified)

