from _skimage2.measure.fit import (
    BaseModel as BaseModel,
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
    _dynamic_max_trials,
)

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
