# Use legacy numpy printing. This fix is made to keep doctests functional.
# For more info, see https://github.com/scikit-image/scikit-image/pull/2935 .
# TODO: remove this workaround once minimal required numpy is set to 1.14.0
from distutils.version import LooseVersion as Version
import numpy as np

if Version(np.__version__) >= Version('1.14'):
    np.set_printoptions(legacy='1.13')

# List of files that pytest should ignore
collect_ignore = ["io/_plugins",]
try:
    import visvis
except ImportError:
    collect_ignore.append("measure/mc_meta/visual_test.py")
