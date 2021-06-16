import importlib
import importlib.util
import os
import sys


def attach(module_name, submodules=None, submod_attrs=None):
    """Attach lazily loaded submodules, functions, or other attributes.

    Typically, modules import submodules and attributes as follows::

      import mysubmodule
      import anothersubmodule

      from .foo import someattr

    The idea is to replace a module's `__getattr__`, `__dir__`, and
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
    module_name : str
        Typically use __name__.
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
            return importlib.import_module(f'{module_name}.{name}')
        elif name in attr_to_modules:
            submod = importlib.import_module(
                f'{module_name}.{attr_to_modules[name]}'
            )
            return getattr(submod, name)
        else:
            raise AttributeError(f'No {module_name} attribute {name}')

    def __dir__():
        return __all__

    if os.environ.get('EAGER_IMPORT', ''):
        for attr in set(attr_to_modules.keys()) | submodules:
            __getattr__(attr)

    return __getattr__, __dir__, list(__all__)


class LazyImportError(ImportError):
    def __init__(self, module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = module


def load(fullname):
    """Return a lazily imported proxy for a module or library.

    We often see the following pattern::

      def myfunc():
          import scipy
          ....

    This is to prevent a library, in this case `scipy`, from being
    imported at function definition time, since that can be slow.

    This function provides a proxy module that, upon access, imports
    the actual module.

    Parameters
    ----------
    fullname : str
        The full name of the package or subpackage to import.  For example::

          sp = lazy_import('scipy')  # import scipy as sp
          spla = lazy_import('scipy.linalg')  # import scipy.linalg as spla

    Returns
    -------
    pm : importlib.util._LazyModule
        Proxy module.  Can be used like any regularly imported module.

    """
    if fullname in sys.modules:
        return sys.modules[fullname]

    spec = importlib.util.find_spec(fullname)
    try:
        module = importlib.util.module_from_spec(spec)
    except:  # noqa: E722
        raise LazyImportError(
            fullname, f'Could not lazy import module {fullname}'
        ) from None
    loader = importlib.util.LazyLoader(spec.loader)

    sys.modules[fullname] = module

    # Make module with proper locking and get it inserted into sys.modules.
    loader.exec_module(module)

    return module
