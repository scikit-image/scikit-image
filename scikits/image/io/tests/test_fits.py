import os.path
import numpy as np
import scikits.image.io as io
from scikits.image import data_dir
import scikits.image.io._plugins.fits_plugin as fplug


import_success = True

try:
    import pyfits
except ImportError:
    import_success = False


def test_fits_plugin_import():
    # Make sure we get an import exception if PyFITS isn't there
    # (not sure how useful this is, but it ensures there isn't some other
    # error when trying to load the plugin)
    try:
        io.use_plugin('fits')
    except ImportError:
        assert import_success == False
    else:
        assert import_success == True

def test_imread_MEF():
    if import_success:
        io.use_plugin('fits')
        testfile = os.path.join(data_dir, 'multi.fits')
        img = io.imread(testfile)
        assert np.all(img==pyfits.getdata(testfile, 1))

def test_imread_simple():
    if import_success:
        io.use_plugin('fits')
        testfile = os.path.join(data_dir, 'simple.fits')
        img = io.imread(testfile)
        assert np.all(img==pyfits.getdata(testfile, 0))

def test_imread_collection_single_MEF():
    if import_success:
        io.use_plugin('fits')
        testfile = os.path.join(data_dir, 'multi.fits')
        ic1 = io.imread_collection(testfile)
        ic2 = io.ImageCollection([(testfile, 1), (testfile, 2), (testfile, 3)],
                  load_func=fplug.FITSFactory)
        assert _same_ImageCollection(ic1, ic2)

def test_imread_collection_MEF_and_simple():
    if import_success:
        io.use_plugin('fits')
        testfile1 = os.path.join(data_dir, 'multi.fits')
        testfile2 = os.path.join(data_dir, 'simple.fits')
        ic1 = io.imread_collection([testfile1, testfile2])
        ic2 = io.ImageCollection([(testfile1, 1), (testfile1, 2),
                                  (testfile1, 3), (testfile2, 0)],
                                 load_func=fplug.FITSFactory)
        assert _same_ImageCollection(ic1, ic2)

def _same_ImageCollection(collection1, collection2):
    """Ancillary function to compare two ImageCollection objects, checking
       that their constituent arrays are equal.
    """
    if len(collection1) != len(collection2):
        return False
    for ext1, ext2 in zip(collection1, collection2):
        if not np.all(ext1 == ext2):
            return False
    return True

