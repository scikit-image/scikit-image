import sys
import importlib
import importlib.util


def install_lazy(module_name, submodules=None, submod_funcs=None):
    """Install lazily loaded submodules and functions.

    Parameters
    ----------
    module_name : str
        Typically use __name__.
    submodules : list
        List of submodules to install.
    submod_funcs : dict
        Dictionary of submodule -> list of functions.  These functions are
        imported as they are used.
    """
    if submod_funcs is None:
        submod_funcs = {}

    if submodules is None:
        submodules = []

    all_funcs = []
    for mod, funcs in submod_funcs.items():
        all_funcs.extend(funcs)

    def __getattr__(name):
        if name in submodules:
            lazy_mod = require(f'skimage.{name}')
            return lazy_mod
        elif name in all_funcs:
            for mod, funcs in submod_funcs.items():
                if name in funcs:
                    submod = importlib.import_module(
                        f'{module_name}.{mod}'
                    )
                    return getattr(submod, name)
        else:
            raise AttributeError(f'No {module_name} attribute {name}')

    def __dir__():
        return submodules + all_funcs

    return __getattr__, __dir__, submodules + all_funcs


def require(fullname):
    if fullname in sys.modules:
        return sys.modules[fullname]

    spec = importlib.util.find_spec(fullname)
    try:
        module = importlib.util.module_from_spec(spec)
    except:
        raise ImportError(f'Could not lazy import module {fullname}') from None
    loader = importlib.util.LazyLoader(spec.loader)

    sys.modules[fullname] = module

    # Make module with proper locking and get it inserted into sys.modules.
    loader.exec_module(module)

    return module
