import os, numpy as np, imghdr
from skimage.util import img_as_ubyte
import skimage.io, skimage.transform

# ==================================================

class Pixel(object):
    """A single pixel in a Picture.

    Attributes
    ----------
    pic : Picture
        The Picture object that this pixel references.
    image : array_like
        Byte array with raw image data (RGB).
    x : int
        Horizontal coordinate of this pixel (left = 0).
    y : int
        Vertical coordinate of this pixel (bottom = 0).
    rgb : tuple
        RGB tuple with red, green, and blue components (0-255)

    """
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
        """Gets the horizontal location (left = 0).

        Returns
        -------
        x : int
            The horizontal location of the pixel.

        """
        return self._x

    @property
    def y(self):
        """Gets the vertical location (bottom = 0).

        Returns
        -------
        y : int
            The vertical location of the pixel.

        """
        return self._y

    @property
    def red(self):
        """Gets the red component of the pixel.

        Returns
        -------
        value : int
            The red component of the pixel (0-255).
        """
        return self._red

    @red.setter
    def red(self, value):
        """Sets the red component of the pixel.

        Parameters
        ----------
        value : int
            Red component of the pixel (0-255)

        """
        self._red = self._validate(value)
        self._setpixel()

    @property
    def green(self):
        """Gets the green component of the pixel.

        Returns
        -------
        value : int
            The green component of the pixel (0-255).
        """
        return self._green

    @green.setter
    def green(self, value):
        """Sets the green component of the pixel.

        Parameters
        ----------
        value : int
            Green component of the pixel (0-255)

        """
        self._green = self._validate(value)
        self._setpixel()

    @property
    def blue(self):
        """Gets the blue component of the pixel.

        Returns
        -------
        value : int
            The blue component of the pixel (0-255).
        """
        return self._blue

    @blue.setter
    def blue(self, value):
        """Sets the blue component of the pixel.

        Parameters
        ----------
        value : int
            Blue component of the pixel (0-255)

        """
        self._blue = self._validate(value)
        self._setpixel()

    @property
    def rgb(self):
        """Gets the RGB color of the pixel.

        Returns
        ----------
        color : tuple
            RGB tuple with red, green, and blue components (0-255)

        """
        return (self.red, self.green, self.blue)

    @rgb.setter
    def rgb(self, value):
        """Sets the RGB color of the pixel.

        Parameters
        ----------
        value : tuple
            RGB tuple with red, green, and blue components (0-255)

        """
        for v in value:
            self._validate(v)

        self._red = value[0]
        self._green = value[1]
        self._blue = value[2]
        self._setpixel()

    def _validate(self, value):
        """Verifies that the pixel value is in [0, 255].

        """
        try:
            value = int(value)
            if (value < 0) or (value > 255):
                raise ValueError()
        except ValueError:
            raise ValueError("Expected an integer between 0 and 255, but got {0} instead!".format(value))

        return value

    def _setpixel(self):
        """Sets the actual pixel value in the picture.
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
    """A group of Pixel objects that can be manipulated together.

    Attributes
    ----------
    pic : Picture
        The Picture object that this pixel group references.
    key : tuple
        tuple with x and y slices or ints for extracting part of raw image data.

    """
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
        """Gets a specific dimension out of the raw image data slice.

        """
        return self._slice[:, :, dim]

    def _setdim(self, dim, value):
        """ Sets a specific dimension in the raw image data slice.

        """
        self._image[self._key[0], self._key[1], dim] = value

    @property
    def red(self):
        """Gets the red component of the pixel.

        Returns
        -------
        value : int
            The red component of the pixel (0-255).
        """
        return self._getdim(0).ravel()

    @red.setter
    def red(self, value):
        """Sets the red component of the pixel.

        Parameters
        ----------
        value : int
            Red component of the pixel (0-255)

        """
        self._setdim(0, value)

    @property
    def green(self):
        """Gets the green component of the pixel.

        Returns
        -------
        value : int
            The green component of the pixel (0-255).
        """
        return self._getdim(1).ravel()

    @green.setter
    def green(self, value):
        """Sets the green component of the pixel.

        Parameters
        ----------
        value : int
            green component of the pixel (0-255)

        """
        self._setdim(1, value)

    @property
    def blue(self):
        """Gets the blue component of the pixel.

        Returns
        -------
        value : int
            The blue component of the pixel (0-255).
        """
        return self._getdim(2).ravel()

    @blue.setter
    def blue(self, value):
        """Sets the blue component of the pixel.

        Parameters
        ----------
        value : int
            blue component of the pixel (0-255)

        """
        self._setdim(2, value)

    @property
    def rgb(self):
        """Gets the RGB color of the pixel.

        Returns
        ----------
        color : tuple
            RGB tuple with red, green, and blue components (0-255)

        """
        return self._getdim(None)

    @rgb.setter
    def rgb(self, value):
        """Sets the RGB color of the pixel.

        Parameters
        ----------
        value : tuple
            RGB tuple with red, green, and blue components (0-255)

        """
        self._setdim(None, value)

    def __iter__(self):
        """Iterates through all pixels in the pixel group.

        """
        x_idx = range(self._pic.width)[self._key[0]]
        y_idx = range(self._pic.height)[self._key[1]]

        for x in x_idx:
            for y in y_idx:
                yield self._pic._makepixel((x, y))

    def __repr__(self):
        return "PixelGroup ({0} pixels)".format(self.size[0] * self.size[1])

# ==================================================

class Picture(object):
    """A 2-D picture made up of pixels.

    Attributes
    ----------
    path : str
        Path to an image file to load.
    size : tuple
        Size of the empty image to create (width, height).
    color : tuple
        Color to fill empty image if size is given (red, green, blue) [0-255].
    image : array_like
        Raw RGB image data [0-255]

    Notes
    -----
    Cannot provide more than one of 'path' and 'size' and 'image'.
    Can only provide 'color' if 'size' provided.

    Examples
    --------
    Load an image from a file

    >>> pic = Picture(path="/path/to/image/file.png")

    Create a blank 100 pixel wide, 200 pixel tall white image

    >>> pic = Picture(size=(100, 200), color=(255, 255, 255))

    Use numpy to make an RGB byte array (shape is height x width x 3)

    >>> import numpy as np
    >>> data = np.zeros(shape=(200, 100, 3), dtype=np.int8)
    >>> data[:, :, 0] = 255  # Set red component to maximum
    >>> pic = Picture(image=data)

    """
    def __init__(self, path=None, size=None, color=None, image=None):
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
        """Creates a Picture from an image file.

        Parameters
        ----------
        path : str
            Path to the image file.

        Returns
        -------
        pic : Picture
            A Picture with the image file data loaded.

        """
        return Picture(path=path)

    @staticmethod
    def from_color(color, size):
        """Creates a single color Picture.

        Parameters
        ----------
        color : tuple
            RGB tuple with the fill color for the picture [0-255]

        size : tuple
            Width and height of the picture in pixels.

        Returns
        -------
        pic : Picture
            A Picture with the given color and size.

        """
        return Picture(color=color, size=size)

    @staticmethod
    def from_image(image):
        """Creates a single color Picture.

        Parameters
        ----------
        color : tuple
            RGB tuple with the fill color for the picture [0-255]

        size : tuple
            Width and height of the picture in pixels.

        Returns
        -------
        pic : Picture
            A Picture with the given color and size.

        """
        return Picture(image=image)

    def save(self, path):
        """Saves the picture to the given path.

        Parameters
        ----------
        path : str
            Path to save the picture (with file extension).

        """
        skimage.io.imsave(path, self._inflate(self._image))
        self._modified = False
        self._path = os.path.abspath(path)
        self._format = imghdr.what(path)

    @property
    def path(self):
        """Gets the path of the picture.

        Returns
        -------
        path : str
            The absolute path of the picture or None if modified.

        """
        return self._path

    @property
    def modified(self):
        """Gets a value indicating if the picture has changed.

        Returns
        -------
        modified : bool
            True if the picture has been changed, False otherwise.

        """
        return self._modified

    @property
    def format(self):
        """Gets the format of the picture.

        Returns
        -------
        format : str
            File format of the picture (e.g., png, jpeg) or None if not saved.

        """
        return self._format

    @property
    def size(self):
        """Gets size of the picture.

        Returns
        -------
        size : tuple
            Width and height of the picture in pixels.

        """
        # skimage dimensions are flipped: y, x
        return (self._image.shape[1], self._image.shape[0])

    @size.setter
    def size(self, value):
        """Sets the size of the picture.

        Parameters
        ----------
        value : tuple
            Desired width and height of the picture. Will resize image as necessary.

        """
        try:
            # Don't resize if no change in size
            if (value[0] != self.width) or (value[1] != self.height):
                # skimage dimensions are flipped: y, x
                self._image = skimage.img_as_ubyte(skimage.transform.resize(self._image,
                    (int(value[1]), int(value[0])), order=0))

                self._modified = True
                self._path = None
        except TypeError:
            raise TypeError("Expected (width, height), but got {0} instead!".format(value))

    @property
    def width(self):
        """Gets the width of the picture.

        Returns
        -------
        width : int
            Width of the picture in pixels.

        """
        return self.size[0]

    @width.setter
    def width(self, value):
        """Sets the width of the picture.

        Parameters
        -------
        value : int
            Desired width of the picture in pixels. Will resize the image as necessary.

        """
        self.size = (value, self.height)

    @property
    def height(self):
        """Gets the height of the picture.

        Returns
        -------
        height : int
            Height of the picture in pixels.

        """
        return self.size[1]

    @height.setter
    def height(self, value):
        """Sets the height of the picture.

        Parameters
        -------
        value : int
            Desired height of the picture in pixels. Will resize the image as necessary.

        """
        self.size = (self.width, value)

    @property
    def inflation(self):
        """Gets the inflation factor.

        Returns
        -------
        inflation : int
            Factor to scale each pixel by when saving or viewing (1 = no inflation).

        """
        return self._inflation

    @inflation.setter
    def inflation(self, value):
        """Sets the inflation factor.

        Parameters
        ----------
        value : int
            Factor to scale each pixel by when saving or viewing (1 = no inflation).

        """
        try:
            value = int(value)
            if value < 1:
                raise ValueError()
            self._inflation = value
        except ValueError:
            raise ValueError("Expected inflation factor to be an integer greater than one")

    def _repr_html_(self):
        return skimage.io.Image(self._inflate(self._image))

    def _repr_png_(self):
        return skimage.io.Image(self._inflate(self._image))

    def show(self):
        """Displays the image in a separate window.

        """
        skimage.io.imshow(self._inflate(self._image))

    def _makepixel(self, xy):
        """ Creates a Pixel object for a given x, y location.

        Parameters
        ----------
        xy : tuple
            Cartesian coordinates to create a Pixel object for.

        """
        # skimage dimensions are flipped: y, x
        rgb = self._image[self.height - xy[1] - 1, xy[0]]
        return Pixel(self, self._image, xy[0], xy[1], rgb)

    def _inflate(self, img):
        """Inflates image according to inflation factor.

        Parameters
        ----------
        img : array_like
            Raw RGB image data to inflate using nearest neighbor algorithm.

        """
        if self._inflation == 1:
            return img

        # skimage dimensions are flipped: y, x
        return img_as_ubyte(skimage.transform.resize(img,
            (self.height * self._inflation,
             self.width * self._inflation), order=0))

    def __iter__(self):
        """Iterates over all pixels in the image.

        """
        for x in xrange(self.width):
            for y in xrange(self.height):
                yield self._makepixel((x, y))

    def __getitem__(self, key):
        """Gets pixels using 2D int or slice notations.

        Parameters
        ----------
        key : tuple
            Horizontal and vertical coordinates or slices.

        Returns
        -------
        p: Pixel or PixelGroup
            Single pixel for coordinates or PixelGroup for slice.

        Examples
        --------
        Get the bottom-left pixel

        >>> pic[0, 0]

        Get the top row of the picture

        >>> pic[:, pic.height-1]

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
        """Sets pixel values using 2D int or slice notations.

        Parameters
        ----------
        key : tuple
            Horizontal and vertical coordinates or slices.
        value : tuple
            RGB tuple with red, green, and blue components [0-255]

        Examples
        --------
        Set the bottom-left pixel to black

        >>> pic[0, 0] = (0, 0, 0)

        Set the top row to red

        >>> pic[:, pic.height-1] = (255, 0, 0)

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

