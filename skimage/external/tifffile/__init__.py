try:
    from tifffile import imread, imsave, TiffFile
except ImportError:
    from .tifffile_local import imread, imsave, TiffFile
