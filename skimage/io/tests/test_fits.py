import os.path
import numpy as np
import skimage.io as io
from skimage import data_dir
from skimage._shared import testing


pyfits_available = True

try:
    from astropy.io import fits as pyfits
except ImportError:
    try:
        import pyfits
    except ImportError:
        pyfits_available = False

if pyfits_available:
    import skimage.io._plugins.fits_plugin as fplug


def test_fits_plugin_import():
    # Make sure we get an import exception if PyFITS isn't there
    # (not sure how useful this is, but it ensures there isn't some other
    # error when trying to load the plugin)
    try:
        io.use_plugin('fits')
    except ImportError:
        assert not pyfits_available
    else:
        assert pyfits_available


def teardown():
    io.reset_plugins()


def _same_ImageCollection(collection1, collection2):
    """
    Ancillary function to compare two ImageCollection objects, checking that
    their constituent arrays are equal.
    """
    if len(collection1) != len(collection2):
        return False
    for ext1, ext2 in zip(collection1, collection2):
        if not np.all(ext1 == ext2):
            return False
    return True
