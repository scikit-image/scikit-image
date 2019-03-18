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
from functools import partial 

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


def _generalized_distance_transform_1d(tuple_arr, dist_func, dist_meet):
    arr = tuple_arr[0]
    if len(tuple_arr) > 1:
        cost_func = lambda x : tuple_arr[1][x]
    else:
        cost_func = lambda x : 0 if arr[x] == 0 else np.finfo(np.float64).max

    rightmost = 0
    length = len(arr)
    domains = np.empty(length+1)
    domains[0] = -np.inf
    domains[1] = np.inf
    centers = np.zeros(length,dtype=int)
    
    for i in range(1,length):
        intersection = dist_meet(i,centers[rightmost],cost_func)
        while intersection <= domains[rightmost]:
            rightmost-=1
            intersection = dist_meet(i,centers[rightmost],cost_func)

        rightmost+=1
        centers[rightmost]=i
        domains[rightmost]=intersection
        domains[rightmost+1] = np.inf

    current_domain = 0
    out = np.empty(length)
    for i in range(length):
        while domains[current_domain+1]<i:
            current_domain += 1
        out[i] = dist_func(i,centers[current_domain],cost_func(centers[current_domain]))
    return out

def generalized_distance_transform(ndarr, cost_func=f, dist_func=euclidean_dist, dist_meet=euclidean_meet):
    gdt1d = partial(_generalized_distance_transform_1d, dist_func=dist_func, dist_meet=dist_meet)
    for dimension in range(ndarr.ndim):
        if dimension == 0:
            out = apply_along_axis(gdt1d, 0, (ndarr,))
        else:
            out = apply_along_axis(gdt1d, dimension, (ndarr, out))
    return out

# try to find a way to generalise the loops
# fix inf ###low priority###
