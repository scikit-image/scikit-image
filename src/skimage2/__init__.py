"""Image Processing for Python (EXPERIMENTAL API version 2)."""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

from _skimage2 import ExperimentalAPIWarning


__all__ = [
    "color",
    "data",
    "draw",
    "exposure",
    "feature",
    "filters",
    "graph",
    "io",
    "measure",
    "metrics",
    "morphology",
    "registration",
    "restoration",
    "segmentation",
    "transform",
    "util",
]


def __getattr__(name):
    if name in __all__:
        submod = importlib.import_module(f"_skimage2.{name}")
        sys.modules[f"skimage2.{name}"] = submod
        globals()[name] = submod
        return submod
    raise AttributeError(f"module 'skimage2' has no attribute '{name}'")


class _Skimage2SubmoduleFinder(importlib.abc.MetaPathFinder):
    """Resolve ``skimage2.<submodule>`` imports to ``_skimage2.<submodule>``.

    Plain attribute access (``import skimage2; skimage2.filters``) already
    works via ``__getattr__`` above, but Python's import statement
    machinery (``from skimage2.filters import gaussian`` or
    ``import skimage2.filters``) bypasses ``__getattr__`` and looks up
    ``skimage2.filters`` directly in ``sys.modules``, where it does not
    exist. See gh-8217.

    This finder registers the real ``_skimage2.<submodule>`` under the
    ``skimage2.<submodule>`` name in ``sys.modules`` the first time it is
    imported, so both import styles work, without eagerly importing every
    public submodule (and their side effects, e.g. ``skimage2.io``
    initializing its plugin system) on a plain ``import skimage2``.

    Uses the modern ``find_spec``/``exec_module`` loader protocol
    (PEP 451); the older ``find_module``/``load_module`` API is no
    longer invoked by Python's import machinery.
    """

    _prefix = "skimage2."

    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith(self._prefix):
            return None
        submod_name = fullname[len(self._prefix) :]
        if "." in submod_name or submod_name not in __all__:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        submod_name = spec.name[len(self._prefix) :]
        mod = importlib.import_module(f"_skimage2.{submod_name}")
        globals()[submod_name] = mod
        return mod

    def exec_module(self, module):
        # The module was already fully executed by importlib.import_module
        # in create_module above; nothing left to do here.
        pass


if not any(isinstance(f, _Skimage2SubmoduleFinder) for f in sys.meta_path):
    sys.meta_path.append(_Skimage2SubmoduleFinder())

warnings.warn(
    "Importing from the `skimage2` namespace is experimental. "
    "Its API is under development and considered unstable!",
    ExperimentalAPIWarning,
    stacklevel=2,
)
