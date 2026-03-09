import sys
from skimage._shared.version_requirements import is_installed

# pyodide maintainers suggest not running tests that use matplotlib,
# https://github.com/pyodide/pyodide-recipes/issues/475#issuecomment-4020982042
has_mpl = "pyodide" not in sys.modules and is_installed("matplotlib", ">=3.3")
