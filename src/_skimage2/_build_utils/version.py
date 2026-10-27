#!/usr/bin/env python3

"""Determine and print version number.

Used in top level ``meson.build``.
"""

import importlib.util
from pathlib import Path


def version():
    """Load the version from ``_skimage2/_version.py``.

    The module is loaded by path, since ``_skimage2`` is not yet built.
    """
    path = Path(__file__).parent / '../../_skimage2/_version.py'
    assert path.is_file()

    spec = importlib.util.spec_from_file_location('_skimage2_version', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__


if __name__ == "__main__":
    print(version())
