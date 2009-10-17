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
import warnings

def _import_opencv_lib(which="cv"):
    """
    Try to import a shared library of OpenCV.

    which - Which library ["cv", "cxcore", "highgui"]
    """
    if sys.platform.startswith("darwin"):
        shared_lib = _tryload_macosx(which)
    elif sys.platform.startswith("linux"):
        shared_lib = ctypes.CDLL('lib' + which + '.so')
    else:
        shared_lib = ctypes.CDLL(which + '.dll')

    if shared_lib is None:
        warnings.warn(RuntimeWarning(
            'The opencv libraries were not found.  Please ensure that they '
            'are installed and available on the system path. '))
    return shared_lib

def _tryload_macosx(which):
    common_paths = [
        '/lib/',
        '/usr/lib/',
        '/usr/local/lib/',
        '/opt/local/lib/', # MacPorts
        '/sw/lib/', # Fink
    ]
    shared_lib = None
    for path in common_paths:
        try:
            libpath =path + "lib" + which + '.dylib' 
            shared_lib = ctypes.CDLL(path + "lib" + which + '.dylib')
            break
        except OSError, e:
            if "image not found" in e.args[0]:
                continue
            raise

    return shared_lib


cv = _import_opencv_lib("cv")
cxcore = _import_opencv_lib("cxcore")
