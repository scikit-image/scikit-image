#!/usr/bin/env python
# encoding: utf-8

"""
This file properly imports the open CV libraries and returns them
as an object. This function goes a longer way to try to find them
since especially on MacOS X Library Paths are not clearly defined.

This module also removes the code duplication in __init__ and
opencv_cv
"""

__all__ = ["cv", "cxcore"]

import ctypes
import sys
import os.path
import warnings

def _import_opencv_lib(which="cv"):
    """
    Try to import a shared library of OpenCV.

    which - Which library ["cv", "cxcore", "highgui"]
    """
    library_paths = ['',
                     '/lib/',
                     '/usr/lib/',
                     '/usr/local/lib/',
                     '/opt/local/lib/', # MacPorts
                     '/sw/lib/', # Fink
                     ]

    if sys.platform.startswith('linux'):
        extensions = ['.so', '.so.1']
    elif sys.platform.startswith("darwin"):
        extension = ['.dylib']
    else:
        extension = ['.dll']
        library_paths = []

    lib = 'lib' + which
    shared_lib = None

    for path in library_paths:
        for ext in extensions:
            try:
                shared_lib = ctypes.CDLL(os.path.join(path, lib + ext))
            except OSError:
                pass
            else:
                return shared_lib

    warnings.warn(RuntimeWarning(
        'The opencv libraries were not found.  Please ensure that they '
        'are installed and available on the system path. '))

cv = _import_opencv_lib("cv")
cxcore = _import_opencv_lib("cxcore")
