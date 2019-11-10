"""
Note about customising distance function:
If dist_func or dist_meet is changed, it is recommended that both of them are changed
as the meeting point of graphs are depended on their equation. Sometimes graphs can be 
parallel and will never touch, ie: |x-1| = |x-2|+10. In that case dist_func needs to  return
np.finfo(np.float64).max if the first graph is at the top and np.finfo(np.float64).min if
it is at the bottom.
"""

import warnings
import numpy as np
from numpy import finfo
from ..util.along_axis import apply_along_axis
from ..util import img_as_float64
from functools import partial 
from ._distance_transform import (_generalized_distance_transform_1d_euclidean,
                                  _generalized_distance_transform_1d_manhattan,
                                  _generalized_distance_transform_1d_slow)

def f(p, posINF=np.inf):
    if p == 0:
        return 0
    return posINF

def euclidean_dist(a,b,c):
    return (a-b)**2+c

def euclidean_meet(a,b,f, posINF=np.inf, negINF=-np.inf):
    out = (f[a]+a**2-f[b]-b**2)/(2*a-2*b)
    return out

def manhattan_dist(a,b,c):
    return np.abs(a-b)+c

def manhattan_meet(a,b,f, posINF=np.inf, negINF=-np.inf):
    s = (a + f[a] + b - f[b]) / 2
    if manhattan_dist(a,s,f[a])==manhattan_dist(b,s,f[b]):
        return s
    s = (a - f[a] + b + f[b]) / 2
    if manhattan_dist(a,s,f[a])==manhattan_dist(b,s,f[b]):
        return s
    if manhattan_dist(a,a,f[a]) > manhattan_dist(b,a,f[b]):
        return posINF
    return -1

def generalized_distance_transform(ndarr_in, func='euclidean', cost_func=f, dist_func=euclidean_dist, dist_meet=euclidean_meet):
    ndarr = ndarr_in.astype(np.double)
    if func == "euclidean":
        gdt1d = partial(_generalized_distance_transform_1d_euclidean, typeINF=float, negINF = finfo(float).min, posINF=finfo(float).max)
    elif func == "manhattan":
        gdt1d = partial(_generalized_distance_transform_1d_manhattan, typeINF=float, negINF = finfo(float).min, posINF=finfo(float).max)
    else:
        gdt1d = partial(_generalized_distance_transform_1d_slow, cost_func=cost_func, dist_func=dist_func, dist_meet=dist_meet)
        warnings.warn("slow")
    output = np.empty(ndarr.shape, dtype=np.double)
    for dimension in range(ndarr.ndim):
        length = ndarr.shape[dimension]
        domains_buffer =np.empty(length+1, dtype=np.double)
        centers_buffer = np.empty(length,dtype=np.intp)
        out_buffer = np.empty(length, dtype=np.double)
        
        if dimension == 0:
            output = apply_along_axis(gdt1d, dimension, (ndarr, output), isfirst=True, domains=domains_buffer, centers=centers_buffer, out=out_buffer)
        else:
            output = apply_along_axis(gdt1d, dimension, (ndarr, output), isfirst=False, domains = domains_buffer, centers = centers_buffer, out = out_buffer)
    
    return output

