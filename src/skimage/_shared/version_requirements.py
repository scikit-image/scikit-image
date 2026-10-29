from _skimage2._shared.version_requirements import (
    get_module as get_module,
    get_module_version as get_module_version,
    is_installed as is_installed,
    require as require,
)  # noqa: F401

__all__ = [
    'get_module',
    'get_module_version',
    'is_installed',
    'require',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
