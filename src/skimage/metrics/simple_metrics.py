from _skimage2.metrics.simple_metrics import (
    peak_signal_noise_ratio as peak_signal_noise_ratio,
    normalized_mutual_information as normalized_mutual_information,
)  # noqa: F401

__all__ = [
    'mean_squared_error',
    'normalized_root_mse',
    'peak_signal_noise_ratio',
    'normalized_mutual_information',
]

import _skimage2.metrics.simple_metrics as ski2_sm

from ..util._backends import dispatchable_shim as _dshim

mean_squared_error = _dshim(ski2_sm.mean_squared_error, module=__name__)
normalized_root_mse = _dshim(ski2_sm.normalized_root_mse, module=__name__)

from skimage._doctest_adapters import adapt_doctests  # noqa: E402

adapt_doctests(globals())
