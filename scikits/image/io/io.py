"""Read and write image files."""

from __future__ import with_statement

__all__ = ['Img', 'MultiImg', 'ImgCollection', 'imread']

from glob import glob
import os.path

import numpy as np
from pil_imread import imread

try:
    from PIL import Image
except ImportError:
    raise ImportError("Could not import the Python Imaging Library (PIL)"
                      " required to load image files.  Please refer to"
                      " http://pypi.python.org/pypi/PIL/ for installation"
                      " instructions.")

#import supreme
#from supreme.lib import EXIF


class Img(np.ndarray):
    """Image data with tags."""

    tags = {'filename' : '',
            'EXIF' : {},
            'info' : {}}

    def __new__(image_cls, arr,  **kwargs):
        """Set the image data and tags according to given parameters.

        Input:
        ------
        `image_cls` : Img class specification
            This is not normally specified by the user.
        `arr` : ndarray
            Image data.
        ``**kwargs`` : Image tags as keywords
            Specified in the form ``tag0=value``, ``tag1=value``.

        """
        x = np.asarray(arr).view(image_cls)
        for tag, value in Img.tags.items():
            setattr(x, tag, kwargs.get(tag, getattr(arr, tag, value)))
        return x

    def __array_finalize__(self, obj):
        """Copy object tags."""
        for tag, value in Img.tags.items():
            setattr(self, tag, getattr(obj, tag, value))
        return

    def __reduce__(self):
        object_state = list(np.ndarray.__reduce__(self))
        subclass_state = {}
        for tag in self.tags:
            subclass_state[tag] = getattr(self, tag)
        object_state[2] = (object_state[2], subclass_state)
        return tuple(object_state)

    def __setstate__(self, state):
        nd_state, subclass_state = state
        np.ndarray.__setstate__(self, nd_state)

        for tag in subclass_state:
            setattr(self, tag, subclass_state[tag])

    #@property
    #def exposure(self):
    #    """Return exposure time based on EXIF tag."""
    #    exposure = self.EXIF['EXIF ExposureTime'].values[0]
    #    return exposure.num / float(exposure.den)


class MultiImg(object):
    """A class containing a single multi-image.

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
    ::
        >>> img = MultiImg(fname)
        >>> img.numframes
        3
        >>> for frame in img:
        ...     frame.shape
        (576, 384)
        (576, 384)
        (576, 384)
    """
    def __init__(self, filename, conserve_memory=True):
        """Load a multi-img"""
        self.filename = filename
        self.conserve_memory = conserve_memory
        self._cached = None

        img = Image.open(self.filename)
        if self.conserve_memory:
            self.numframes = self._find_numframes(img)
        else:
            self._frames = self._getallframes(img)
            self.numframes = len(self._frames)

    def _find_numframes(self, img):
        """Find the number of frames in the multi-img."""
        try:
            i = 0
            while True:
                i += 1
                img.seek(i)
        except EOFError:
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
        numframes = self.numframes
        if -numframes <= n < numframes:
            n = n% numframes
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
        return self.numframes

    def __str__(self):
        return str(self.filename)


class ImgCollection(object):
    """Load and manage a collection of images."""

    def __init__(self, file_pattern, conserve_memory=True, grey=False):
        """Load image files.

        Note that files are always stored in alphabetical order.

        Input:
        ------
        file_pattern : string
            Path and pattern of files to load, e.g. ``data/*.jpg``.
        conserve_memory : bool
            If True, never keep more than one in memory at a specific
            time.  Otherwise, images will be cached once they are loaded.
        grey : bool
            If True, convert the input images to grey-scale.

        Example:
        --------
        >>> from os.path import dirname, join
        >>> data_dir = join(dirname(__file__), 'tests')

        >>> c = ImgCollection(data_dir + '/*.png')
        >>> len(c)
        3
        >>> c[2].shape
        (20, 20, 3)

        """
        self.files = sorted(glob(file_pattern))

        if conserve_memory:
            memory_slots = 1
        else:
            memory_slots = len(self.files)

        self.conserve_memory = conserve_memory
        self.grey = grey
        self.data = np.empty(memory_slots, dtype=object)

    def __getitem__(self, n, _cached=np.array(-1)):
        """Return image n in the queue.

        Loading is done on demand.

        Input:
        ------
        n : int
            Number of image required.

        Output:
        -------
        img : array
           Image #n in the collection.

        """
        idx = n % len(self.data)
        if (_cached != n and self.conserve_memory) or (self.data[idx] is None):
            image_data = imread(self.files[n], self.grey)

            #with file(self.files[n]) as f:
                #exif = EXIF.process_file(f)

            #self.data[idx] = Img(image_data,
                                   #filename=os.path.basename(self.files[n]),
                                   #EXIF=exif, info={})

        _cached.flat = n

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
