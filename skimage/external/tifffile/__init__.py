try:
    from tifffile import *
except ImportError:
    from ._tifffile import *
