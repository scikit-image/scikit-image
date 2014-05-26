from ._ccomp import label as _label

def label(input, neighbors=8, background=None, return_num=False):
    return _label(input, neighbors, background, return_num)

label.__doc__ = _label.__doc__
