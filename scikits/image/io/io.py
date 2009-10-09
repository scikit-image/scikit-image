"""Read and write image files."""

from __future__ import with_statement

__all__ = ['Image','ImageCollection','imread']

from glob import glob
import os.path

import numpy as N
from numpy.testing import set_local_path, restore_path
from scipy.misc.pilutil import imread

set_local_path('../..')
import supreme
from supreme.lib import EXIF
restore_path()

class Image(N.ndarray):
    """Image data with tags."""

    tags = {'filename' : '',
            'EXIF' : {},
            'info' : {}}

    def __new__(image_cls,arr,**kwargs):
        """Set the image data and tags according to given parameters.

        Input:
        ------
        `image_cls` : Image class specification
            This is not normally specified by the user.
        `arr` : ndarray
            Image data.
        ``**kwargs`` : Image tags as keywords
            Specified in the form ``tag0=value``, ``tag1=value``.

        """
        x = N.asarray(arr).view(image_cls)
        for tag,value in Image.tags.items():
            setattr(x,tag,kwargs.get(tag,getattr(arr,tag,value)))
        return x

    def __array_finalize__(self, obj):
        """Copy object tags."""
        for tag,value in Image.tags.items():
            setattr(self,tag,getattr(obj,tag,value))
        return

    def __reduce__(self):
        object_state = list(N.ndarray.__reduce__(self))
        subclass_state = {}
        for tag in self.tags:
            subclass_state[tag] = getattr(self,tag)
        object_state[2] = (object_state[2],subclass_state)
        return tuple(object_state)

    def __setstate__(self,state):
        nd_state,subclass_state = state
        N.ndarray.__setstate__(self,nd_state)

        for tag in subclass_state:
            setattr(self,tag,subclass_state[tag])

    @property
    def exposure(self):
        """Return exposure time based on EXIF tag."""
        exposure = self.EXIF['EXIF ExposureTime'].values[0]
        return exposure.num / float(exposure.den)

class ImageCollection(object):
    """Load and manage a collection of images."""

    def __init__(self,file_pattern,conserve_memory=True,grey=False):
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
        >>> data_dir = join(dirname(__file__),'tests')

        >>> c = ImageCollection(data_dir + '/*.png')
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
        self.data = N.empty(memory_slots,dtype=object)

    def __getitem__(self,n,_cached=N.array(-1)):
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
            image_data = imread(self.files[n],self.grey)

            with file(self.files[n]) as f:
                exif = EXIF.process_file(f)

            self.data[idx] = Image(image_data,
                                   filename=os.path.basename(self.files[n]),
                                   EXIF=exif,info={})

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
