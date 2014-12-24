from ._ccomp import label as _label

def label(input, neighbors=8, background=None, return_num=False,
        connectivity=None):
    return _label(input, neighbors, background, return_num, connectivity)

label.__doc__ = _label.__doc__
