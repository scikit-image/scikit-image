import numpy as np

def one_d(arr, f):
    INF = np.inf
    k=0
    n=len(arr)
    z = (n+1)*[None]
    z[0] = -INF
    z[1] = INF
    v = [None]*n
    v[0]=0
    
    for q in range(1,n):
        s = (f(q) + q**2 - f(v[k]) - (v[k])**2) / (2*q-2*v[k])
        while s <= z[k]:
            k-=1
            s = (f(q) + q**2 - f(v[k]) - (v[k])**2) / (2*q-2*v[k])

        k+=1
        v[k]=q
        z[k]=s
        z[k+1] = INF

    k=0
    d = n*[None]
    for q in range(n):
        while z[k+1]<q:
            k+=1
        d[q] = (q-v[k])**2+f(v[k])
    return d


def generalized_distance_transform(ndarr, f=False):
    if f == False:
        f = lambda p : 0 if p == 0 else np.finfo(np.float64).max
    
    if len(ndarr.shape)==1:
        one_d(ndarr,f)
    shape = ndarr.shape
    mut_shape = list(shape)
    out = np.zeros(shape)

    for i in range(len(shape)):
        changed_shape = mut_shape[:]
        missing = changed_shape.pop(i)
        if i == 0:
            nd_recursion(f, ndarr, changed_shape, out, 0, i, missing, [])
        else:
            out2 = np.zeros(shape)
            nd_recursion(f, ndarr, changed_shape, out2, 0, i, missing, [], pre=out)
            out = out2
    return out

def nd_recursion(f, ndarr, shape, out, c, i, missing, temp, pre=False):
    if c <= len(shape)-1:
        for j in range(shape[c]):
                nd_recursion(f, ndarr, shape, out, c+1, i, missing, temp+[j], pre=pre)
    else:
        temp.insert(i, range(missing))
        temp = tuple(temp)
        #print(temp)
        if isinstance(pre,bool):
            f2 = lambda x: f((ndarr[temp])[x])
        else:
            f2 = lambda x: (pre[temp])[x]
        out[temp] = one_d(ndarr[temp], f2)

def f(p):
    if p == 0:
        return 0
    return np.finfo(np.float64).max
