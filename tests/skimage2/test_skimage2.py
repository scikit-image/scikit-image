import os
import sys
import subprocess

import pytest

from _skimage2._shared._dependency_checks import is_wasm


@pytest.mark.skipif(is_wasm, reason="emscripten does not support processes")
def test_import_skimage2_warning():
    result = subprocess.run(
        [sys.executable, "-c", "import skimage2"],
        capture_output=True,
        text=True,
    )
    assert result.stderr.startswith(
        "<string>:1: "
        "ExperimentalAPIWarning: "
        "Importing from the `skimage2` namespace is experimental"
    )
    assert result.stdout == ""
    assert result.returncode == 0


@pytest.mark.skipif(is_wasm, reason="emscripten does not support processes")
@pytest.mark.parametrize("namespace", ["skimage2", "_skimage2"])
def test_no_eager_skimage_import(namespace):
    """Test that importing `_skimage2` doesn't import `skimage` eagerly."""

    # Print all imported modules starting with "skimage" after importing `_skimage2`
    test_code = (
        f"import {namespace}\n"
        "import sys\n"
        "for module in sys.modules.keys():\n"
        "    if 'skimage' in module:\n"
        "        print(module)\n"
    )
    # Use subprocess to sidestep import state in the current interpreter
    env = os.environ.copy()
    env["EAGER_IMPORT"] = "true"
    result = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0

    imported_modules = result.stdout.splitlines()

    assert namespace in imported_modules
    # `EAGER_IMPORT=true` should have triggered lazy import of submodules
    assert len(imported_modules) > 1
    # `skimage` should *not* be in triggered imports.
    # `skimage2` and `_skimage2` (imported by the first) are acceptable here.
    for module in imported_modules:
        top_module, *_ = module.partition(".")
        assert top_module != "skimage"
        assert "skimage" in top_module
