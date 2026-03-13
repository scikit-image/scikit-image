import os
import sys
import subprocess

import pytest
import importlib
from pathlib import Path

import skimage2
from skimage._shared._dependency_checks import is_wasm


@pytest.mark.thread_unsafe(reason="importlib.reload is not thread-safe")
def test_import_skimage2_warning():
    regex = "Importing from the `skimage2` namespace is experimental"
    with pytest.warns(UserWarning, match=regex) as record:
        importlib.reload(skimage2)
    assert len(record) == 1
    assert record[0].category == skimage2.ExperimentalAPIWarning
    # `importlib.reload` adds a stacklevel, so we actually want the warning to
    # be raised in importlib and not in this file (compare OS-agnostic paths)
    warning_path = Path(record[0].filename)
    assert warning_path.parts[-2:] == ("importlib", "__init__.py")


@pytest.mark.skipif(is_wasm, reason="emscripten does not support processes")
def test_no_eager_skimage_import():
    """Test that importing `skimage2` doesn't import `skimage` eagerly."""

    # Print all imported modules starting with "skimage" after importing `skimage2`
    test_code = (
        "import skimage2\n"
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

    assert result.stderr == (
        "<string>:1: ExperimentalAPIWarning: "
        "Importing from the `skimage2` namespace is experimental. "
        "Its API is under development and considered unstable!\n"
    )
    assert result.returncode == 0

    imported_modules = result.stdout.splitlines()

    # `EAGER_IMPORT=true` should have triggered more imports than just `skimage2`
    assert len(imported_modules) > 1
    # All triggered imports should be `skimage2`, *not* `skimage`
    for module in imported_modules:
        assert module.startswith("skimage2")
