"""Image Processing SciKit (Toolbox for SciPy)

``scikits-image`` (a.k.a. ``skimage``) is a collection of algorithms for image
processing and computer vision.

The main package of ``skimage`` just provides a few utilities for converting
between image data types; for most features, you'll need to import one of the
subpackages described below:

Subpackages
-----------
color
    Utilities for converting between color spaces.
data
    Image data for testing and examples.
draw
    Functions for drawing on images.
exposure
    Utilities for adjusting the image intensity.
feature
    Functions for detecting features in images (e.g. texture, corners, etc.).
filter
    Image filters for denoising, sharpening, edge-finding, and more.
graph
    Functions based on graph-theoretic representations of images.
io
    Utilities for reading, saving, and displaying images and video.
measure
    Functions for image measurement.
morphology
    Mathematical morphology operations on images.
segmentation
    Algorithms segmenting images into regions.
transform
    Transform images into domains that are useful for detection and analysis.
util
    Utilities for image data-type conversion.

"""

import os.path as _osp

pkg_dir = _osp.abspath(_osp.dirname(__file__))
data_dir = _osp.join(pkg_dir, 'data')

from version import version as __version__

def _setup_test(verbose=False):
    import gzip
    import functools

    args = ['', '--exe', '-w', pkg_dir]
    if verbose:
        args.extend(['-v', '-s'])

    try:
        import nose as _nose
    except ImportError:
        print("Could not load nose.  Unit tests not available.")
        return None
    else:
        f = functools.partial(_nose.run, 'skimage', argv=args)
        f.__doc__ = 'Invoke the skimage test suite.'
        return f

test = _setup_test()
if test is None:
    try:
        del test
    except NameError:
        pass

test_verbose = _setup_test(verbose=True)
if test_verbose is None:
    try:
        del test
    except NameError:
        pass

def get_log(name):
    """Return a console logger.

    Output may be sent to the logger using the `debug`, `info`, `warning`,
    `error` and `critical` methods.

    Parameters
    ----------
    name : str
        Name of the log.

    References
    ----------
    .. [1] Logging facility for Python,
           http://docs.python.org/library/logging.html

    """
    import logging, sys
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
    return logging.getLogger(name)

from .util.dtype import *
