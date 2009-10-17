from opencv_constants import *

# Note: users should be able to import this module even if
# the extensions are uncompiled or the opencv libraries unavailable.
# In that case, the opencv functionality is simply unavailable.

loaded = False

try:
    from opencv_cv import *
except ImportError:
    print """*** The opencv extension was not compiled.  Run

python setup.py build_ext -i

in the source directory to build in-place.  Please refer to INSTALL.txt
for further detail."""
except RuntimeError:
    # Libraries could not be loaded
    print "*** Skipping import of OpenCV functions."
    del (opencv_backend, opencv_cv)
else:
    loaded = True

del opencv_constants

