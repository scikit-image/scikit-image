import ctypes

# try to open the opencv libs
# raise an exception if the libs are not found

# linux
try:
    ctypes.CDLL('libcv.so')    
except:
    # windows
    try:
        ctypes.CDLL('cv.dll')
    except:        
        raise RuntimeError('The opencv libraries were not found. Please make sure they are installed and available on the system path.')

from opencv_constants import *
from opencv_cv import *

#def test(level=1, verbosity=1):
#    from numpy.testing import Tester
#    return Tester().test(level, verbosity)

