import os
import sys
import subprocess

import pytest

from _skimage2._shared._dependency_checks import is_wasm


@pytest.mark.thread_unsafe(reason="importlib.reload is not thread-safe")
@pytest.mark.skipif(is_wasm, reason="emscripten does not support processes")
def test_import_skimage2_warning():
    result = subprocess.run(
        [sys.executable, "-c", "import skimage2"],
        capture_output=True,
        text=True,
    )
    assert (
        "ExperimentalAPIWarning: "
        "Importing from the `skimage2` namespace is experimental" in result.stderr
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


def test_skimage2_submodule_imports():
    """Test that public submodules can be imported from skimage2."""
    import _skimage2
    
    with pytest.warns(_skimage2.ExperimentalAPIWarning):
        import skimage2

    # Check a few public submodules
    from skimage2.filters import gaussian
    from _skimage2.filters import gaussian as _gaussian

    assert gaussian is _gaussian

    from skimage2.morphology import disk
    from _skimage2.morphology import disk as _disk

    assert disk is _disk

    from skimage2.color import rgb2gray
    from _skimage2.color import rgb2gray as _rgb2gray

    assert rgb2gray is _rgb2gray

    # Private submodules should not be accessible
    with pytest.raises(ImportError):
        import skimage2._shared

    with pytest.raises(ImportError):
        import skimage2._build_utils

    with pytest.raises(ImportError):
        import skimage2._vendored

    # Existing top-level imports should still work
    assert hasattr(skimage2, "filters")
    assert hasattr(skimage2.filters, "gaussian")

