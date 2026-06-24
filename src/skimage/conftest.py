"""
Pytest configuration for ``skimage`` doctests.

* ``handle_np2`` sets NumPy legacy print options (needed once public doctests
  grow beyond the small shim-only set).
* ``skimage_doctest_namespace`` exposes implementation-module globals so
  shim doctests can use names such as ``np`` defined in ``_skimage2``.
"""

import builtins
import importlib

import pytest

from _skimage2.conftest import handle_np2 as handle_np2  # noqa: F401

__all__ = ['handle_np2']

_BUILTIN_NAMES = frozenset({'abs', 'all', 'any', 'len', 'max', 'min', 'range', 'sum'})


def _skimage2_impl_module(nodeid: str) -> str | None:
    """Return ``_skimage2`` module name for a ``skimage`` doctest nodeid."""
    module_path, _, _ = nodeid.partition('::')
    module_path = module_path.replace('\\', '/')
    if '/skimage/' not in module_path or not module_path.endswith('.py'):
        return None
    rel = module_path.split('/skimage/', 1)[-1].removesuffix('.py').replace('/', '.')
    return f'_skimage2.{rel}'


@pytest.fixture(autouse=True)
def skimage_doctest_namespace(doctest_namespace, request):
    """Merge ``_skimage2`` module globals into shim doctest namespaces."""
    impl_name = _skimage2_impl_module(request.node.nodeid)
    if impl_name is None:
        return

    try:
        impl = importlib.import_module(impl_name)
    except ModuleNotFoundError:
        return

    for key, value in impl.__dict__.items():
        if not key.startswith('__'):
            doctest_namespace.setdefault(key, value)

    doctest_namespace['__file__'] = impl.__file__

    for name in _BUILTIN_NAMES:
        doctest_namespace[name] = getattr(builtins, name)
