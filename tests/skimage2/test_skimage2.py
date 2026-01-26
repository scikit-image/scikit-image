import pytest
import importlib
from pathlib import Path

import skimage2


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
