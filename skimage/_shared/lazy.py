import importlib
import importlib.util
import os
import sys
import ast


def attach(package_name, submodules=None, submod_attrs=None):
    """Attach lazily loaded submodules, functions, or other attributes.

    Typically, modules import submodules and attributes as follows::

      import mysubmodule
      import anothersubmodule

      from .foo import someattr

    The idea is to replace a package's `__getattr__`, `__dir__`, and
    `__all__`, such that all imports work exactly the way they did
    before, except that they are only imported when used.

    The typical way to call this function, replacing the above imports, is::

      __getattr__, __lazy_dir__, __all__ = lazy.attach(
        __name__,
        ['mysubmodule', 'anothersubmodule'],
        {'foo': 'someattr'}
      )

    This functionality requires Python 3.7 or higher.

    Parameters
    ----------
    package_name : str
        Typically use ``__name__``.
    submodules : set
        List of submodules to attach.
    submod_attrs : dict
        Dictionary of submodule -> list of attributes / functions.
        These attributes are imported as they are used.

    Returns
    -------
    __getattr__, __dir__, __all__

    """
    if submod_attrs is None:
        submod_attrs = {}

    if submodules is None:
        submodules = set()
    else:
        submodules = set(submodules)

    attr_to_modules = {
        attr: mod for mod, attrs in submod_attrs.items() for attr in attrs
    }

    __all__ = list(submodules | attr_to_modules.keys())

    def __getattr__(name):
        if name in submodules:
            return importlib.import_module(f'{package_name}.{name}')
        elif name in attr_to_modules:
            submod = importlib.import_module(
                f'{package_name}.{attr_to_modules[name]}'
            )
            return getattr(submod, name)
        else:
            raise AttributeError(f'No {package_name} attribute {name}')

    def __dir__():
        return __all__

    eager_import = os.environ.get('EAGER_IMPORT', '')
    if eager_import not in ['', '0', 'false']:
        for attr in set(attr_to_modules.keys()) | submodules:
            __getattr__(attr)

    return __getattr__, __dir__, list(__all__)


def load(fullname):
    """Return a lazily imported proxy for a module.

    We often see the following pattern::

      def myfunc():
          import scipy as sp
          sp.argmin(...)
          ....

    This is to prevent a module, in this case `scipy`, from being
    imported at function definition time, since that can be slow.

    This function provides a proxy module that, upon access, imports
    the actual module.  So the idiom equivalent to the above example is::

      sp = lazy.load("scipy")

      def myfunc():
          sp.argmin(...)
          ....

    The initial import time is fast because the actual import is delayed
    until the first attribute is requested. The overall import time may
    decrease as well for users that don't make use of large portions
    of the library.

    Parameters
    ----------
    fullname : str
        The full name of the module or submodule to import.  For example::

          sp = lazy.load('scipy')  # import scipy as sp
          spla = lazy.load('scipy.linalg')  # import scipy.linalg as spla

    Returns
    -------
    pm : importlib.util._LazyModule
        Proxy module.  Can be used like any regularly imported module.
        Actual loading of the module occurs upon first attribute request.

    """
    try:
        return sys.modules[fullname]
    except KeyError:
        pass

    spec = importlib.util.find_spec(fullname)
    if spec is None:
        raise ModuleNotFoundError(f"No module name '{fullname}'")

    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module

    loader = importlib.util.LazyLoader(spec.loader)
    loader.exec_module(module)

    return module


class _StubVisitor(ast.NodeVisitor):
    """AST visitor to parse a stub file for submodules and submod_attrs."""

    def __init__(self):
        self._submodules = set()
        self._submod_attrs = {}

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.level != 1:
            raise ValueError(
                "Only within-module imports are supported (`from .* import`)"
            )
        if node.module:
            attrs: list = self._submod_attrs.setdefault(node.module, [])
            attrs.extend(alias.name for alias in node.names)
        else:
            self._submodules.update(alias.name for alias in node.names)


def attach_stub(package_name: str, filename: str):
    """Attach lazily loaded submodules, functions from a type stub.

    This is a variant on ``attach`` that will parse a `.pyi` stub file to
    infer ``submodules`` and ``submod_attrs``. This allows static type
    checkers to find imports, while still providing lazy loading at runtime.

    Parameters
    ----------
    package_name : str
        Typically use ``__name__``.
    filename : str
        Path to `.py` file which has an adjacent `.pyi` file.
        Typically use ``__file__``.

    Returns
    -------
    __getattr__, __dir__, __all__
        The same output as ``attach``.

    Raises
    ------
    ValueError
        If a stub file is not found for `filename`, or if the stubfile is
        formmated incorrectly (e.g. if it contains an relative import from
        outside of the module)

    Examples
    --------
    >>> from .._shared import lazy
    >>> __getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
    """
    stubfile = filename if filename.endswith("i") else f"{filename}i"

    if not os.path.exists(stubfile):
        raise ValueError(
            f"Cannot load imports from non-existent stub {stubfile!r}"
        )

    with open(stubfile) as f:
        stub_node = ast.parse(f.read())

    visitor = _StubVisitor()
    visitor.visit(stub_node)
    return attach(package_name, visitor._submodules, visitor._submod_attrs)
