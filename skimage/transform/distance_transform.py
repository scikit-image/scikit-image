"""
Note about customising distance function:
If dist_func or dist_meet is changed, it is recommended that both of them are changed
as the meeting point of graphs are depended on their equation. Sometimes graphs can be 
parallel and will never touch, ie: |x-1| = |x-2|+10. In that case dist_func needs to  return
np.finfo(np.float64).max if the first graph is at the top and np.finfo(np.float64).min if
it is at the bottom.
"""


import numpy as np
from skimage.util.along_axis import apply_along_axis
import warnings

def f(p):
    if p == 0:
        return 0
    return np.finfo(np.float64).max

def euclidean_dist(a,b,c):
    return (a-b)**2+c

def euclidean_meet(a,b,f):
    return (f(a)+a**2-f(b)-b**2)/(2*a-2*b)

def manhattan_dist(a,b,c):
    return np.abs(a-b)+c

def manhattan_meet(a,b,f):
    s = (a + f(a) + b - f(b)) / 2
    if manhattan_dist(a,s,f(a))==manhattan_dist(b,s,f(b)):
        return s
    s = (a - f(a) + b + f(b)) / 2
    if manhattan_dist(a,s,f(a))==manhattan_dist(b,s,f(b)):
        return s
    if manhattan_dist(a,a,f(a)) > manhattan_dist(b,a,f(b)):
        return np.finfo(np.float64).max
    return np.finfo(np.float64).min


def one_d(tuple_arr, dist_func=euclidean_dist, dist_meet=euclidean_meet):
    arr = tuple_arr[0]
    if len(tuple_arr) > 1:
        f = lambda x : tuple_arr[1][x]
    else:
        f = lambda x : 0 if arr[x] == 0 else np.finfo(np.float64).max

    k=0
    n=len(arr)
    z = np.empty(n+1)
    z[0] = -np.inf
    z[1] = np.inf
    v = [None]*n
    v[0]=0
    
    for q in range(1,n):
        s = dist_meet(q,v[k],f)
        while s <= z[k]:
            k-=1
            s = dist_meet(q,v[k],f)

        k+=1
        v[k]=q
        z[k]=s
        z[k+1] = np.inf

    k=0
    d = np.empty(n)
    for q in range(n):
        while z[k+1]<q:
            k+=1
        d[q] = dist_func(q,v[k],f(v[k]))
    return d

def generalized_distance_transform(ndarr, f=f, dist_func=euclidean_dist, dist_meet=euclidean_meet):
    for i in range(ndarr.ndim):
        if i == 0:
            out = apply_along_axis(one_d, 0, (ndarr,))
        else:
            out = apply_along_axis(one_d, i, (ndarr, out))
    return out

# new np.applyalongaxis ###higher priority###
# convert remaining lists to arrays
# try to find a way to generalise the loops
# low priority : make temp less awkeward
# fix inf ###high priority###
# itertools/np.applyalongaxis
