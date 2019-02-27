"""
Note about customising distance function:

If dist_func or dist_meet is changed, it is recommended that both of them are changed
as the meeting point of graphs are depended on their equation. Sometimes graphs can be 
parallel and will never touch, ie: |x-1| = |x-2|+10. In that case dist_func needs to  return
np.finfo(np.float64).max if the first graph is at the top and np.finfo(np.float64).min if
it is at the bottom.

"""


import numpy as np


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

def one_d(arr, f, dist_func=euclidean_dist, dist_meet=euclidean_meet):
    INF = np.inf
    k=0
    n=len(arr)
    z = (n+1)*[None]
    z[0] = -INF
    z[1] = INF
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
        z[k+1] = INF

    k=0
    d = np.empty(n)
    for q in range(n):
        while z[k+1]<q:
            k+=1
        d[q] = dist_func(q,v[k],f(v[k]))
    return d


def generalized_distance_transform(ndarr, f=f, dist_func=euclidean_dist, dist_meet=euclidean_meet):
    if len(ndarr.shape)==1:
        one_d(ndarr,f)
    shape = ndarr.shape
    mut_shape = list(shape)
    out = np.zeros(shape)

    for i in range(ndarr.ndim):
        changed_shape = mut_shape[:]
        missing = changed_shape.pop(i)

        def nd_recursion(c, temp, pre=False):
            if c < len(changed_shape):
                for j in range(changed_shape[c]):
                    nd_recursion(c+1, temp+[j], pre=pre)
            else:
                temp.insert(i, range(missing))
                temp = tuple(temp)
                if isinstance(pre,bool):
                    f2 = lambda x: f((ndarr[temp])[x])
                else:
                    f2 = lambda x: (pre[temp])[x]
                out[temp] = one_d(ndarr[temp], f2, dist_func, dist_meet)


        if i == 0:
            nd_recursion(0, [])
            out2 = out
        else:
            out2 = np.zeros(shape)
            out, out2 = out2, out
            nd_recursion(0, [], pre=out2)
    return out
