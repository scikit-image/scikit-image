# Adapted from https://stackoverflow.com/a/51126745/214686

import sys
import importlib
import importlib.util


def install_lazy(module_name, submodules):
    def __getattr__(name):
        if name in submodules:
            lazy_mod = require(f'skimage.{name}')
            return lazy_mod
        else:
            raise AttributeError(f'No skimage attribute {name}')


    def __dir__():
        return submodules

    return __getattr__, __dir__


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
