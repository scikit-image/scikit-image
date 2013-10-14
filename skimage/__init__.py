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
import imp as _imp
import functools as _functools
from skimage._shared.utils import deprecated as _deprecated

pkg_dir = _osp.abspath(_osp.dirname(__file__))
data_dir = _osp.join(pkg_dir, 'data')

try:
    from .version import version as __version__
except ImportError:
    __version__ = "unbuilt-dev"
del version


try:
    _imp.find_module('nose')
except ImportError:
    def _test(verbose=False):
        """This would invoke the skimage test suite, but nose couldn't be
        imported so the test suite can not run.
        """
        raise ImportError("Could not load nose. Unit tests not available.")
else:
    def _test(verbose=False):
        """Invoke the skimage test suite."""
        import nose
        args = ['', pkg_dir, '--exe']
        if verbose:
            args.extend(['-v', '-s'])
        nose.run('skimage', argv=args)

# do not use `test` as function name as this leads to a recursion problem with
# the nose test suite
test = _test
test_verbose = _functools.partial(test, verbose=True)
test_verbose.__doc__ = test.__doc__


class _Log(Warning):
    pass


class _FakeLog(object):
    def __init__(self, name):
        """
        Parameters
        ----------
        name : str
            Name of the log.
        repeat : bool
            Whether to print repeating messages more than once (False by
            default).
        """
        self._name = name

        import warnings
        warnings.simplefilter("always", _Log)

        self._warnings = warnings

    def _warn(self, msg, wtype):
        self._warnings.warn('%s: %s' % (wtype, msg), _Log)

    def debug(self, msg):
        self._warn(msg, 'DEBUG')

    def info(self, msg):
        self._warn(msg, 'INFO')

    def warning(self, msg):
        self._warn(msg, 'WARNING')

    warn = warning

    def error(self, msg):
        self._warn(msg, 'ERROR')

    def critical(self, msg):
        self._warn(msg, 'CRITICAL')

    def addHandler(*args):
        pass

    def setLevel(*args):
        pass


@_deprecated()
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
    if name is None:
        name = 'skimage'
    else:
        name = 'skimage.' + name

    return _FakeLog(name)


from .util.dtype import *
