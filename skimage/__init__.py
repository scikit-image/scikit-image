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
    Drawing primitives (lines, text, etc.) that operate on NumPy arrays.
exposure
    Image intensity adjustment, e.g., histogram equalization, etc.
feature
    Feature detection and extraction, e.g., texture analysis corners, etc.
filters
    Sharpening, edge finding, rank filters, thresholding, etc.
graph
    Graph-theoretic operations, e.g., shortest paths.
io
    Reading, saving, and displaying images and video.
measure
    Measurement of image properties, e.g., similarity and contours.
morphology
    Morphological operations, e.g., opening or skeletonization.
novice
    Simplified interface for teaching purposes.
restoration
    Restoration algorithms, e.g., deconvolution algorithms, denoising, etc.
segmentation
    Partitioning an image into multiple regions.
transform
    Geometric and other transforms, e.g., rotation or the Radon transform.
util
    Generic utilities.
viewer
    A simple graphical user interface for visualizing results and exploring
    parameters.

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

import os.path as osp
import imp
import functools
import warnings
import sys

pkg_dir = osp.abspath(osp.dirname(__file__))
data_dir = osp.join(pkg_dir, 'data')

__version__ = '0.15.dev0'


if sys.version_info < (3,):
    raise ImportError("""

You are running scikit-image on Python 2.

Unfortunately, scikit-image 0.15 and above no longer work on this
version of Python.  You therefore have two options: either upgrade to
Python 3, or install an older version of scikit-image using

 $ pip install 'scikit-image<0.15'

Please also consider updating `pip` and `setuptools`:

 $ pip install pip setuptools --upgrade

Newer versions of these tools avoid installing packages incompatible
with your version of Python.
""")


try:
    imp.find_module('pytest')
except ImportError:
    def _test(doctest=False, verbose=False):
        """This would run all unit tests, but pytest couldn't be
        imported so the test suite can not run.
        """
        raise ImportError("Could not load pytest. Unit tests not available.")

else:
    def _test(doctest=False, verbose=False):
        """Run all unit tests."""
        import pytest
        import warnings
        args = ['--pyargs', 'skimage']
        if verbose:
            args.extend(['-v', '-s'])
        if doctest:
            args.extend(['--doctest-modules'])
            # Make sure warnings do not break the doc tests
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                success = pytest.main(args)
        else:
            success = pytest.main(args)
        # Return sys.exit code
        if success:
            return 0
        else:
            return 1


# do not use `test` as function name as this leads to a recursion problem with
# the nose test suite
test = _test
test_verbose = functools.partial(test, verbose=True)
test_verbose.__doc__ = test.__doc__
doctest = functools.partial(test, doctest=True)
doctest.__doc__ = doctest.__doc__
doctest_verbose = functools.partial(test, doctest=True, verbose=True)
doctest_verbose.__doc__ = doctest.__doc__

from .util.dtype import *


def lookfor(what):
    """Do a keyword search on scikit-image docstrings.

    Parameters
    ----------
    what : str
        Words to look for.

    """
    import numpy as np
    import sys
    return np.lookfor(what, sys.modules[__name__])


del warnings, functools, osp, imp, sys
