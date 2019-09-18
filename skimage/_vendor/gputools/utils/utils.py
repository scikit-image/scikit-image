from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import math

def pad_to_shape(d, dshape, mode = "constant"):
    """
    pad array d to shape dshape
    """
    if d.shape == dshape:
        return d

    diff = np.array(dshape)- np.array(d.shape)
    #first shrink
    slices  = tuple(slice(-x//2,x//2) if x<0 else slice(None,None) for x in diff)
    res = d[slices]
    #then pad
    # return np.pad(res,[(n/2,n-n/2) if n>0 else (0,0) for n in diff],mode=mode)
    return np.pad(res,[(int(np.ceil(d/2.)),d-int(np.ceil(d/2.))) if d>0 else (0,0) for d in diff],mode=mode)



def _is_power2(n):
    return next_power_of_2(n) == n

def next_power_of_2(n):
    return 1 if n == 0 else 2**math.ceil(math.log2(n))

def pad_to_power2(data, axis = None, mode="constant"):
    """
    pad data to a shape of power 2
    if axis == None all axis are padded
    """
    if axis is None:
        axis = list(range(data.ndim))

    if np.all([_is_power2(n) for i, n in enumerate(data.shape) if i in axis]):
        return data
    else:
        return pad_to_shape(data, [(next_power_of_2(n) if i in axis else n) for i, n in enumerate(data.shape)], mode)




def get_cache_dir():
    import sys, os
    import appdirs
    return os.path.join(appdirs.user_cache_dir("pyopencl", "pyopencl"),
                "pyopencl-compiler-cache-v2-py%s" % (
                    ".".join(str(i) for i in sys.version_info),))
    # from tempfile import gettempdir
    # import getpass
    # import os
    # import sys
    #
    # return os.path.join(gettempdir(),
    #                 "pyopencl-compiler-cache-v2-uid%s-py%s" % (
    #                     getpass.getuser(), ".".join(str(i) for i in sys.version_info)))

def remove_cache_dir():
    import shutil

    cache_dir = get_cache_dir()
    print(("try removing cache dir: %s"%cache_dir))
    try:
        shutil.rmtree(cache_dir)
    except Exception as e:
        print(e)



