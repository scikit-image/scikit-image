try:
    from tifffile import *
except ImportError:
    from .tifffile_local import *
