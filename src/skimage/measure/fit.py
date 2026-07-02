from warnings import warn

from _skimage2.measure.fit import (
    CircleModel as CircleModel,
    EllipseModel as EllipseModel,
    LineModelND as LineModelND,
    RansacModelProtocol as RansacModelProtocol,
    add_from_estimate as add_from_estimate,
    ransac as ransac,
)  # noqa: F401


__all__ = [
    'BaseModel',
    'CircleModel',
    'EllipseModel',
    'LineModelND',
    'RansacModelProtocol',
    'add_from_estimate',
    'ransac',
]

from _skimage2.measure.fit import (  # noqa: F401
    _PARAMS_DEP_START,
    _PARAMS_DEP_STOP,
    _dynamic_max_trials,
)

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())


class BaseModel:
    def __init_subclass__(self):
        warn(
            f'`BaseModel` deprecated since version {_PARAMS_DEP_START} and '
            f'will be removed in version {_PARAMS_DEP_STOP}',
            category=FutureWarning,
            stacklevel=2,
        )
