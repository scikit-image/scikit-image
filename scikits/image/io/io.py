"""Read and write image files."""

from __future__ import with_statement

__all__ = ['MultiImage', 'ImageCollection', 'imread']

from glob import glob
import os.path

import numpy as np
from pil_imread import imread

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
    >>> img = MultiImage(fname)  #doctest: +SKIP
    >>> len(img)
    3
    >>> for frame in img:
    ...     print frame.shape
    (576, 384)
    (576, 384)
    (576, 384)
    """
    def __init__(self, filename, conserve_memory=True):
        """Load a multi-img"""
        self._filename = filename
        self.conserve_memory = conserve_memory
        self._cached = None

        img = Image.open(self._filename)
        if self.conserve_memory:
            self._numframes = self._find_numframes(img)
        else:
            self._frames = self._getallframes(img)
            self._numframes = len(self._frames)

    @property
    def filename(self):
        return self._filename

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
        return np.asarray(img)

    def _getallframes(self, img):
        """Extract all frames from the multi-img."""
        frames = []
        try:
            i = 0
            while True:
                frames.append(np.asarray(img))
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
    """Load and manage a collection of images."""

    def __init__(self, file_pattern, conserve_memory=True, as_grey=False):
        """Load image files.

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

        Attributes
        ----------
        files : list of str
            A list of files in the collection, ordered alphabetically.
        as_grey : bool
            Whether images are converted to grey-scale.

        Examples
        --------
        >>> from os.path import dirname, join  #doctest: +SKIP
        >>> data_dir = join(dirname(__file__), 'tests')

        >>> c = ImageCollection(data_dir + '/*.png')
        >>> len(c)
        3
        >>> c[2].shape
        (20, 20, 3)
        """
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

        self.conserve_memory = conserve_memory
        self._cached = None
        #TODO: make as_grey a property, or provide a setter that reloads data
        self.as_grey = as_grey
        self.data = np.empty(memory_slots, dtype=object)

    @property
    def files(self):
        return self._files

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
        num = len(self.files)
        if -num <= n < num:
            n = n % num
        else:
            raise IndexError, "There are only %s images in the collection"%num
        idx = n % len(self.data)

        print num, idx, n
        if (self.conserve_memory and n != self._cached) or (self.data[idx] is None):
            self.data[idx] = imread(self.files[n], self.as_grey)

        return self.data[idx]

    def __iter__(self):
        """Iterate over the images."""
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        """Number of images in collection."""
        return len(self.files)

    def __str__(self):
        return str(self.files)
