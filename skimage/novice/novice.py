import os
from PIL import Image
from itertools import islice, product
from io import BytesIO

# ================================================== 

class pixel(object):
    def __init__(self, pic, data, x, y, rgb):
        self.__picture = pic
        self.__data = data
        self.__x = x
        self.__y = y
        self.__red = self.__validate(rgb[0])
        self.__green = self.__validate(rgb[1])
        self.__blue = self.__validate(rgb[2])

    @property
    def x(self):
        """Gets the horizontal location (left = 0)"""
        return self.__x

    @property
    def y(self):
        """Gets the vertical location (bottom = 0)"""
        return self.__y

    @property
    def red(self):
        """Gets or sets the red component"""
        return self.__red

    @red.setter
    def red(self, value):
        self.__red = self.__validate(value)
        self.__setpixel()

    @property
    def green(self):
        """Gets or sets the green component"""
        return self.__green

    @green.setter
    def green(self, value):
        self.__green = self.__validate(value)
        self.__setpixel()

    @property
    def blue(self):
        """Gets or sets the blue component"""
        return self.__blue

    @blue.setter
    def blue(self, value):
        self.__blue = self.__validate(value)
        self.__setpixel()

    @property
    def rgb(self):
        return (self.red, self.green, self.blue)

    @rgb.setter
    def rgb(self, value):
        """Gets or sets the color with an (r, g, b) tuple"""
        for v in value:
            self.__validate(v)

        self.__red = value[0]
        self.__green = value[1]
        self.__blue = value[2]
        self.__setpixel()

    def __validate(self, value):
        """Verifies that the pixel value is in [0, 255]"""
        try:
            value = int(value)
            if (value < 0) or (value > 255):
                raise ValueError()
        except ValueError:
            raise ValueError("Expected an integer between 0 and 255, but got {0} instead!".format(value))

        return value

    def __setpixel(self):
        """
        Sets the actual pixel value in the picture.
        NOTE: Using Cartesian coordinate system!
        """
        self.__data[self.__x, self.__picture.height - self.__y - 1] = \
                (self.red, self.green, self.blue)

        # Modified pictures lose their paths
        self.__picture._picture__path = None
        self.__picture._picture__modified = True

    def __repr__(self):
        return "pixel (red: {0}, green: {1}, blue: {2})".format(self.red, self.green, self.blue)

# ================================================== 

class pixelgroup(object):
    def __init__(self, pixels):
        self.__pixels = pixels

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
        return iter(self.__pixels)

    def __repr__(self):
        return "pixelgroup ({0} pixels)".format(len(self.__pixels))

# ================================================== 

class picture(object):
    def __init__(self, path):
        self.__path = path
        image = Image.open(path)
        self.__format = image.format

        # We convert the image to RGB automatically so
        # (r, g, b) tuples can be used everywhere.
        self.__image = image.convert("RGB")
        self.__data = self.__image.load()
        self.__modified = False

        self.__inflation = 1

    def save(self, path):
        """Saves the picture to the given path."""
        self.__inflate(self.__image).save(path)
        self.__modified = False
        self.__path = os.path.abspath(path)

        # Need to re-open the image to get the format
        # for some reason (likely because we converted to RGB).
        self.__format = Image.open(path).format

    @property
    def path(self):
        """Gets the path of the picture"""
        return self.__path

    @property
    def modified(self):
        """Gets a value indicating if the picture has changed"""
        return self.__modified

    @property
    def format(self):
        """Gets the format of the picture (e.g., PNG)"""
        return self.__format

    @property
    def size(self):
        """Gets or sets the size of the picture with a (width, height) tuple"""
        return self.__image.size

    @size.setter
    def size(self, value):
        try:
            # Don't resize if no change in size
            if (value[0] != self.width) or (value[1] != self.height):
                self.__image = self.__image.resize(value)
                self.__data = self.__image.load()
                self.__modified = True
                self.__path = None
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
        return self.__inflation

    @inflation.setter
    def inflation(self, value):
        try:
            value = int(value)
            if value < 0:
                raise ValueError()
            self.__inflation = value
        except ValueError:
            raise ValueError("Expected inflation factor to be an integer greater than zero")

    def show(self):
        """Returns an IPython image of the picture for display in an IPython notebook"""
        from IPython.core.display import Image as IPImage

        if self.path and self.inflation == 1:
            # Picture hasn't been modified, just use the file directly
            return IPImage(filename=self.path)
        else:
            # Convert picture to in-memory PNG
            data = BytesIO()
            self.__inflate(self.__image).save(data, format="png")
            data.seek(0)
            return IPImage(data=data.getvalue(), format="png", embed=True)

    def __makepixel(self, xy):
        """
        Creates a pixel object for a given x, y location.
        NOTE: Using Cartesian coordinate system!
        """
        (x,y) = xy
        rgb = self.__data[x, self.height - y - 1]
        return pixel(self, self.__data, x, y, rgb)

    def __inflate(self, img):
        """Returns resized image using inflation factor (nearest neighbor)"""
        return img.resize((self.width * self.__inflation,
            self.height * self.__inflation), Image.NEAREST)

    def __negidx(self, idx, bound):
        """Handles negative indices by wrapping around bound"""
        if (idx is None) or idx >= 0:
            return idx
        else:
            return bound + idx

    def __iter__(self):
        """Iterates over all pixels in the image"""
        for x in xrange(self.width):
            for y in xrange(self.height):
                yield self.__makepixel((x, y))

    def __keys(self, key):
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
                    slx = self.__negidx(slx, self.width)
                    if (slx < 0) or (slx >= self.width):
                        raise IndexError("Index out of range")

                if isinstance(sly, int):
                    sly = self.__negidx(sly, self.height)
                    if (sly < 0) or (sly >= self.height):
                        raise IndexError("Index out of range")

                # self[x, y]
                if isinstance(slx, int) and isinstance(sly, int):
                    return (slx, sly)
                
                if isinstance(slx, int):
                    slx = [slx]
                else: # slice (allow negative indices)
                    start = self.__negidx(slx.start, self.width)
                    stop  = self.__negidx(slx.stop,  self.width)
                    slx = islice(xrange(self.width), start, stop, slx.step)
                    
                if isinstance(sly, int):
                    sly = [sly]
                else: # slice (allow negative indices)
                    start = self.__negidx(sly.start, self.height)
                    stop  = self.__negidx(sly.stop,  self.height)
                    sly = islice(xrange(self.height), start, stop, sly.step)

                return product(slx, sly)

        # if either left or right is not an int or a slice, or
        # if the key is not a pair, fall through
        raise TypeError("Invalid key type")

    def __getitem__(self, key):
        """Gets pixels using 2D int or slice notations"""
        keys = self.__keys(key)
        if isinstance(keys, tuple):
            return self.__makepixel(keys)
        else:
            return pixelgroup(map(self.__makepixel, keys))

    def __setitem__(self, key, value):
        """Sets pixelvalues using 2D int or slice notations"""
        keys = self.__keys(key)
        if isinstance(keys, tuple):
            pixel = self[keys[0], keys[1]]
            pixel.rgb = value
        else:
            for (x,y) in keys:
                pixel = self[x,y]
                pixel.rgb = value

    def __repr__(self):
        return "picture (format: {0}, path: {1}, modified: {2})".format(self.format, self.path, self.modified)

