from _skimage2.metrics.simple_metrics import *  # noqa: F403
from _skimage2.metrics.simple_metrics import __doc__  # noqa: F401
import _skimage2.metrics.simple_metrics as ski2_sm

from ..util._backends import dispatchable_shim as _dshim

mean_squared_error = _dshim(ski2_sm.mean_squared_error, module=__name__)
normalized_root_mse = _dshim(ski2_sm.normalized_root_mse, module=__name__)
