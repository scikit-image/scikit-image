"""
lazy_loader makes it easy to load subpackages and functions on demand.

"""
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
