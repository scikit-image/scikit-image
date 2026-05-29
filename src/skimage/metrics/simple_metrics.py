from _skimage2.metrics.simple_metrics import *  # noqa: F403
from _skimage2.metrics.simple_metrics import __doc__  # noqa: F401
from ..util._backends import dispatchable

mean_squared_error = dispatchable(mean_squared_error)
normalized_root_mse = dispatchable(normalized_root_mse)
