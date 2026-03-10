import sys
import pytest
from skimage._shared.version_requirements import is_installed

# pyodide maintainers suggest not running tests that use matplotlib,
# https://github.com/pyodide/pyodide-recipes/issues/475#issuecomment-4020982042
is_pyodide = "pyodide" in sys.modules
has_mpl = is_installed("matplotlib", ">=3.3")


def uses_matplotlib(test_func):
    if is_pyodide:
        return pytest.mark.skip(reason=(
            "importing matplotlib fails randomly. "
            "See https://github.com/pyodide/pyodide-recipes/issues/475"
        ))(test_func)
    if not has_mpl:
        return pytest.mark.skip(reason="matplotlib not installed")(test_func)
    return test_func
