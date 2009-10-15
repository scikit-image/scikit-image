import ctypes
import warnings

from opencv_constants import *

libs_found = True

try:
    from opencv_cv import *
except:
    warnings.warn(RuntimeWarning(
        'The opencv libraries were not found.  Please ensure that they '
        'are installed and available on the system path. '
        '*** Skipping import of OpenCV functions.'))
    libs_found = False
