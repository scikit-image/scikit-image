from ._ccomp import label as _label

def label(input, neighbors=None, background=None, return_num=False,
        connectivity=None):
    return _label(input, neighbors, background, return_num, connectivity)

label.__doc__ = _label.__doc__
