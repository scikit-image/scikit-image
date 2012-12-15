"""Image Processing SciKit (Toolbox for SciPy)

``scikit-image`` (a.k.a. ``skimage``) is a collection of algorithms for image
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
import imp
import functools

pkg_dir = _osp.abspath(_osp.dirname(__file__))
data_dir = _osp.join(pkg_dir, 'data')

try:
    from .version import version as __version__
except ImportError:
    __version__ = "unbuilt-dev"


try:
    imp.find_module('nose')
except ImportError:
    def test(verbose=False):
        """This would invoke the skimage test suite, but nose couldn't be
        imported so the test suite can not run.
        """
        raise ImportError("Could not load nose.  Unit tests not available.")
else:
    def test(verbose=False):
        """Invoke the skimage test suite."""
        import nose
        args = ['', pkg_dir, '--exe']
        if verbose:
            args.extend(['-v', '-s'])
            nose.run('skimage', argv=args)

test_verbose = functools.partial(test, verbose=True)
test_verbose.__doc__ = test.__doc__


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

    formatter = logging.Formatter(
        '%(name)s: %(levelname)s: %(message)s'
    )

    try:
        handler = logging.StreamHandler(stream=sys.stdout)
    except TypeError:
        handler = logging.StreamHandler(strm=sys.stdout)
    handler.setFormatter(formatter)

    log = get_log()
    log.addHandler(handler)
    log.setLevel(logging.WARNING)
    log.propagate = False

_setup_log()

from .util.dtype import *