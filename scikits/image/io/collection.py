"""Data structures to hold collections of images, with optional caching."""

from __future__ import with_statement

__all__ = ['MultiImage', 'ImageCollection', 'imread']

from glob import glob
import os.path

import numpy as np
from io import imread

from PIL import Image


class MultiImage(object):
    """A class containing a single multi-frame image.

    Parameters
    ----------
    filename : str
        The complete path to the image file.
    conserve_memory : bool, optional
        Whether to conserve memory by only caching a single frame. Default is
        True.
    dtype : dtype, optional
        NumPy data-type specifier. If given, the returned image has this type.
        If None (default), the data-type is determined automatically.

    Attributes
    ----------
    filename : str
        The complete path to the image file.
    conserve_memory : bool
        Whether memory is conserved by only caching a single frame.
    numframes : int
        The number of frames in the image.

    Notes
    -----
    If ``conserve_memory=True`` the memory footprint can be reduced, however
    the performance can be affected because frames have to be read from file
    more often.

    The last accessed frame is cached, all other frames will have to be read
    from file.

    Examples
    --------
    >>> import os.path
    >>> fname = os.path.join('tests', 'data', 'multipage.tif')

    >>> img = MultiImage(fname)
    >>> len(img)
    2
    >>> for frame in img:
    ...     print frame.shape
    (15, 10)
    (15, 10)

    The two frames in this image can be shown with matplotlib:

    .. plot:: show_collection.py
    """
    def __init__(self, filename, conserve_memory=True, dtype=None):
        """Load a multi-img."""
        self._filename = filename
        self._conserve_memory = conserve_memory
        self._dtype = dtype
        self._cached = None

        img = Image.open(self._filename)
        if self._conserve_memory:
            self._numframes = self._find_numframes(img)
        else:
            self._frames = self._getallframes(img)
            self._numframes = len(self._frames)

    @property
    def filename(self):
        return self._filename

    @property
    def conserve_memory(self):
        return self._conserve_memory

    def _find_numframes(self, img):
        """Find the number of frames in the multi-img."""
        i = 0
        while True:
            i += 1
            try:
                img.seek(i)
            except EOFError:
                break
        return i

    def _getframe(self, framenum):
        """Open the image and extract the frame."""
        img = Image.open(self.filename)
        img.seek(framenum)
        return np.asarray(img, dtype=self._dtype)

    def _getallframes(self, img):
        """Extract all frames from the multi-img."""
        frames = []
        try:
            i = 0
            while True:
                frames.append(np.asarray(img, dtype=self._dtype))
                i += 1
                img.seek(i)
        except EOFError:
            return frames

    def __getitem__(self, n):
        """Return the n-th frame as an array.

        Parameters
        ----------
        n : int
            Number of the required frame.

        Returns
        -------
        frame : ndarray
           The n-th frame.
        """
        numframes = self._numframes
        if -numframes <= n < numframes:
            n = n % numframes
        else:
            raise IndexError, "There are only %s frames in the image"%numframes

        if self.conserve_memory:
            if not self._cached == n:
                frame = self._getframe(n)
                self._cached = n
                self._cachedframe = frame
            return self._cachedframe
        else:
            return self._frames[n]

    def __iter__(self):
        """Iterate over the frames."""
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        """Number of images in collection."""
        return self._numframes

    def __str__(self):
        return str(self.filename) + ' [%s frames]'%self._numframes


class ImageCollection(object):
    """Load and manage a collection of image files.

    Note that files are always stored in alphabetical order.

    Parameters
    ----------
    file_pattern : str or list of str
        Path(s) and pattern(s) of files to load. The path can be absolute
        or relative. If given as a list of strings, each string in the list
        is a separate pattern. Files are found by passing the pattern(s) to
        the ``glob.glob`` function.
    conserve_memory : bool, optional
        If True, never keep more than one in memory at a specific
        time.  Otherwise, images will be cached once they are loaded.
    as_grey : bool, optional
        If True, convert the input images to grey-scale. This does not
        affect images that are already in a grey-scale format.
    dtype : dtype, optional
        NumPy data-type specifier. If given, the returned image has this type.
        If None (default), the data-type is determined automatically.

    Attributes
    ----------
    files : list of str
        A list of files in the collection, ordered alphabetically.
    as_grey : bool
        Whether images are converted to grey-scale.

    Examples
    --------
    >>> from scikits.image.io import io
    >>> from scikits.image import data_dir

    >>> coll = io.ImageCollection(data_dir + '/*.png')
    >>> len(coll)
    2
    >>> coll.files
    ['.../scikits/image/data/camera.png', .../scikits/image/data/color.png']
    >>> coll[0].shape
    (256, 256)

    When `as_grey` is changed, a color image is returned in grey-scale:

    >>> coll[1].shape
    (370, 371, 3)
    >>> coll.as_grey = True
    >>> coll[1].shape
    (256, 256)
    """
    def __init__(self, file_pattern, conserve_memory=True, as_grey=False,
                 dtype=None):
        """Load and manage a collection of images."""
        if isinstance(file_pattern, basestring):
            self._files = sorted(glob(file_pattern))
        elif isinstance(file_pattern, list):
            self._files = []
            for pattern in file_pattern:
                self._files.extend(glob(pattern))
            self._files.sort()

        if conserve_memory:
            memory_slots = 1
        else:
            memory_slots = len(self._files)

        self._conserve_memory = conserve_memory
        self._cached = None
        self._as_grey = as_grey
        self._dtype = dtype
        self.data = np.empty(memory_slots, dtype=object)

    @property
    def files(self):
        return self._files

    @property
    def as_grey(self):
        """Whether images are converted to grey-scale.

        If this property is changed, all images in memory get reloaded.
        """
        return self._as_grey

    @as_grey.setter
    def as_grey(self, newgrey):
        if not newgrey == self._as_grey:
            self._as_grey = newgrey
            self.reload()

    @property
    def conserve_memory(self):
        return self._conserve_memory

    def __getitem__(self, n):
        """Return image n in the collection.

        Loading is done on demand.

        Parameters
        ----------
        n : int
            The image number to be returned.

        Returns
        -------
        img : ndarray
           The `n`-th image in the collection.
        """
        n = self._check_imgnum(n)
        idx = n % len(self.data)

        if (self.conserve_memory and n != self._cached) or \
               (self.data[idx] is None):
            self.data[idx] = imread(self.files[n], self.as_grey,
                                    dtype=self._dtype)
            self._cached = n

        return self.data[idx]

    def _check_imgnum(self, n):
        """Check that the given image number is valid."""
        num = len(self.files)
        if -num <= n < num:
            n = n % num
        else:
            raise IndexError, "There are only %s images in the collection"%num
        return n

    def __iter__(self):
        """Iterate over the images."""
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        """Number of images in collection."""
        return len(self.files)

    def __str__(self):
        return str(self.files)

    def reload(self, n=None):
        """Reload one or more images from file.

        Parameters
        ----------
        n : None or int
            The number of the image to reload. If None (default), all images in
            memory are reloaded.  If `n` specifies an image not yet in memory,
            it is loaded.

        Returns
        -------
        None

        Notes
        -----
        `reload` is used to reload all images in memory when `as_grey` is
        changed.
        """
        if n is not None:
            n = self._check_numimg(n)
            idx = n % len(self.data)
            self.data[idx] = imread(self.files[n], self.as_grey,
                                    dtype=self._dtype)
        else:
            for idx, img in enumerate(self.data):
                if img is not None:
                    self.data[idx] = imread(self.files[idx], self.as_grey,
                                            dtype=self._dtype)
