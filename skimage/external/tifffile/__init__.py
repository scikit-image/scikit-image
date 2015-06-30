try:
    from tifffile import imread, imsave, TiffFile
except ImportError:
    from .tifffile_local import imread, imsave, TiffFile
    import tifffile_local, _tifffile
    assert tifffile_local.decodelzw == _tifffile.decodelzw, \
        "The _tifffile.so extension module could not be loaded."
