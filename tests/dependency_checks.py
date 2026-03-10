import sys
import pytest
from skimage._shared.version_requirements import is_installed

# pyodide maintainers suggest not running tests that use matplotlib,
# https://github.com/pyodide/pyodide-recipes/issues/475#issuecomment-4020982042
is_pyodide = "pyodide" in sys.modules
has_mpl = is_installed("matplotlib", ">=3.3")

uses_matplotlib = pytest.mark.skipif(not has_mpl, reason="matplotlib not installed")(
    pytest.mark.skipif(
        is_pyodide,
        reason=(
            "importing matplotlib fails randomly. "
            "See https://github.com/pyodide/pyodide-recipes/issues/475"
        ),
    )
)
