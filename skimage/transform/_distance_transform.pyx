import numpy as np
cimport numpy as np

cdef f(bint p):
    cdef double out
    if p == 0:
        out = 0
    out = np.finfo(np.float64).max
    return out

cdef euclidean_dist(double a, double b, double c):
    cdef double out = (a-b)**2+c
    return out

cdef euclidean_meet(double a, double b, np.ndarray[double, ndim=1] f):
    cdef double out = (f[a]+a**2-f[b]-b**2)/(2*a-2*b)
    return out

cdef manhattan_dist(double a, double b, double c):
    cdef double out = (a-b)**2+c
    return out

cdef manhattan_meet(double a, double b, np.ndarray[double, ndim=1] f):
    cdef double s
    s = (a + f[a] + b - f[b]) / 2
    if manhattan_dist(a,s,f[a])==manhattan_dist(b,s,f[b]):
        return s
    s = (a - f[a] + b + f[b]) / 2
    if manhattan_dist(a,s,f[a])==manhattan_dist(b,s,f[b]):
        return s
    if manhattan_dist(a,a,f[a]) > manhattan_dist(b,a,f[b]):
        return np.finfo(np.float64).max
    return np.finfo(np.float64).min

cdef _generalized_distance_transform_1d_euclidean(double[:] arr, double[:] cost_arr,
                                       bint isfirst, np.ndarray[double, ndim=1] domains,
                                       np.ndarray[int, ndim=1] centers, np.ndarray[double, ndim=1] out):
    cdef short length = len(arr)
    cdef short i, rightmost, current_domain
    cdef double intersection

    if isfirst:
        for i in range(length):
            cost_arr[i] = f(<bint>arr[i])

    rightmost = 0
    domains[0] = -np.inf
    domains[1] = np.inf
    
    for i in range(1,length):
        intersection = euclidean_meet(i,centers[rightmost],cost_arr)
        while intersection <= domains[rightmost]:
            rightmost-=1
            intersection = euclidean_meet(i,centers[rightmost],cost_arr)

        rightmost+=1
        centers[rightmost]=i
        domains[rightmost]=intersection
        domains[rightmost+1] = np.inf

    current_domain = 0

    for i in range(length):
        while domains[current_domain+1]<i:
            current_domain += 1
        out[i] = euclidean_dist(i,centers[current_domain],cost_arr[centers[current_domain]])
    return out

cdef _generalized_distance_transform_1d_manhattan(double[:] arr, double[:] cost_arr,
                                       bint isfirst, np.ndarray[double, ndim=1] domains,
                                       np.ndarray[int, ndim=1] centers, np.ndarray[double, ndim=1] out):
    cdef short length = len(arr)
    cdef short i, rightmost, current_domain
    cdef double intersection

    if isfirst:
        for i in range(length):
            cost_arr[i] = f(<bint>arr[i])

    rightmost = 0
    domains[0] = -np.inf
    domains[1] = np.inf
    
    for i in range(1,length):
        intersection = manhattan_meet(i,centers[rightmost],cost_arr)
        while intersection <= domains[rightmost]:
            rightmost-=1
            intersection = manhattan_meet(i,centers[rightmost],cost_arr)

        rightmost+=1
        centers[rightmost]=i
        domains[rightmost]=intersection
        domains[rightmost+1] = np.inf

    current_domain = 0

    for i in range(length):
        while domains[current_domain+1]<i:
            current_domain += 1
        out[i] = manhattan_dist(i,centers[current_domain],cost_arr[centers[current_domain]])
    return out

cdef _generalized_distance_transform_1d_slow(double[:] arr, double[:] cost_arr,
                                       cost_func, dist_func, dist_meet,
                                       bint isfirst, np.ndarray[double, ndim=1] domains,
                                       np.ndarray[int, ndim=1] centers, np.ndarray[double, ndim=1] out):
    cdef short length = len(arr)
    cdef short i, rightmost, current_domain
    cdef double intersection

    if isfirst:
        for i in range(length):
            cost_arr[i] = cost_func(<bint>arr[i])

    rightmost = 0
    domains[0] = -np.inf
    domains[1] = np.inf
    
    for i in range(1,length):
        intersection = dist_meet(i,centers[rightmost],cost_arr)
        while intersection <= domains[rightmost]:
            rightmost-=1
            intersection = dist_meet(i,centers[rightmost],cost_arr)

        rightmost+=1
        centers[rightmost]=i
        domains[rightmost]=intersection
        domains[rightmost+1] = np.inf

    current_domain = 0

    for i in range(length):
        while domains[current_domain+1]<i:
            current_domain += 1
        out[i] = dist_func(i,centers[current_domain],cost_arr[centers[current_domain]])
    return out