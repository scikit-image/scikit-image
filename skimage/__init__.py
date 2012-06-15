"""Image Processing SciKit (Toolbox for SciPy)

``scikits-image`` (a.k.a. ``skimage``) is a collection of algorithms for image
processing and computer vision.

The main package of ``skimage`` only provides a few utilities for converting
between image data types; for most features, you need to import one of the
following subpackages:

Subpackages
-----------
color
    Color space conversion.
data
    Test images and example data.
draw
    Image drawing primitives (lines, text, etc.).
exposure
    Image intensity adjustment (e.g., histogram equalization).
feature
    Feature detection (e.g. texture analysis, corners, etc.).
filter
    Sharpening, edge finding, denoising, etc.
graph
    Graph-theoretic operations, e.g. dynamic programming (shortest paths).
io
    Reading, saving, and displaying images and video.
measure
    Measurement of image properties, e.g., similarity and contours.
morphology
    Morphological operations, e.g. opening or skeletonization.
segmentation
    Splitting an image into self-similar regions.
transform
    Geometric and other transforms, e.g. rotation or the Radon transform.
util
    Generic utilities.

Utility Functions
-----------------
get_log
    Returns the ``skimage`` log.  Use this to print debug output.
img_as_float
    Convert an image to floating point format, with values in [0, 1].
img_as_uint
    Convert an image to unsigned integer format, with values in [0, 65535].
img_as_int
    Convert an image to signed integer format, with values in [-32768, 32767].
img_as_ubyte
    Convert an image to unsigned byte format, with values in [0, 255].

"""

import os.path as _osp

pkg_dir = _osp.abspath(_osp.dirname(__file__))
data_dir = _osp.join(pkg_dir, 'data')

try:
    from .version import version as __version__
except ImportError:
    __version__ = "unbuilt-dev"


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


def get_log(name=None):
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
    import logging

    if name is None:
        name = 'skimage'
    else:
        name = 'skimage.' + name

    log = logging.getLogger(name)
    return log


def _setup_log():
    """Configure root logger.

    """
    import logging
    import sys

    log = logging.getLogger()

    try:
        handler = logging.StreamHandler(stream=sys.stdout)
    except TypeError:
        handler = logging.StreamHandler(strm=sys.stdout)

    formatter = logging.Formatter(
        '%(name)s: %(levelname)s: %(message)s'
        )
    handler.setFormatter(formatter)

    log.addHandler(handler)
    log.setLevel(logging.WARNING)

_setup_log()

from .util.dtype import *
