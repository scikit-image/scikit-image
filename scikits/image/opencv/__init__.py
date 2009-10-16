import ctypes
import sys

# try to open the opencv libs
# prints a warning if the libs are not found
from _libimport import cv, cxcore

from opencv_constants import *
from opencv_cv import *
