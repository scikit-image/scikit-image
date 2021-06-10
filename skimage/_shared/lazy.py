import importlib
import importlib.util
import os


def install_lazy(module_name, submodules=None, submod_attrs=None):
    """Install lazily loaded submodules, and functions or other attributes.

    Parameters
    ----------
    module_name : str
        Typically use __name__.
    submodules : set
        List of submodules to install.
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

    if os.environ.get('EAGER_IMPORT', None):
        for attr in set(attr_to_modules.keys()) | submodules:
            __getattr__(attr)

    return __getattr__, __dir__, list(__all__)
