from opencv_constants import *

from .. import get_log as _get_log

_log = _get_log("scikits.image.opencv")
_log.warn("""
The scikits.image OpenCV wrappers will be removed in the next release.

These wrappers were written before OpenCV's own allowed manipulation
of NumPy arrays without copying.  Since they now do, please switch to
the official bindings::

  http://opencv.willowgarage.com/wiki/""")

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

