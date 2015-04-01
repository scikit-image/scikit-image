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

import os.path as _osp
import imp as _imp
import functools as _functools
import warnings as _warnings

pkg_dir = _osp.abspath(_osp.dirname(__file__))
data_dir = _osp.join(pkg_dir, 'data')

__version__ = '0.11.3'

try:
    _imp.find_module('nose')
except ImportError:
    def _test(doctest=False, verbose=False):
        """This would run all unit tests, but nose couldn't be
        imported so the test suite can not run.
        """
        raise ImportError("Could not load nose. Unit tests not available.")

else:
    def _test(doctest=False, verbose=False):
        """Run all unit tests."""
        import nose
        args = ['', pkg_dir, '--exe', '--ignore-files=^_test']
        if verbose:
            args.extend(['-v', '-s'])
        if doctest:
            args.extend(['--with-doctest', '--ignore-files=^\.',
                         '--ignore-files=^setup\.py$$', '--ignore-files=test'])
            # Make sure warnings do not break the doc tests
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                success = nose.run('skimage', argv=args)
        else:
            success = nose.run('skimage', argv=args)
        # Return sys.exit code
        if success:
            return 0
        else:
            return 1


# do not use `test` as function name as this leads to a recursion problem with
# the nose test suite
test = _test
test_verbose = _functools.partial(test, verbose=True)
test_verbose.__doc__ = test.__doc__
doctest = _functools.partial(test, doctest=True)
doctest.__doc__ = doctest.__doc__
doctest_verbose = _functools.partial(test, doctest=True, verbose=True)
doctest_verbose.__doc__ = doctest.__doc__

from .util.dtype import *
