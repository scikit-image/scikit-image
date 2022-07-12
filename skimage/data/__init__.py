from .._shared import lazy

# add exports to the __init__.pyi file adjacent to this file
__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
