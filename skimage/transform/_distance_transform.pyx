#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
cimport numpy as np
from libc.stdint cimport uint16_t
from numpy cimport ndarray
from numpy.math cimport INFINITY
"""
Implementation choices
- Inline had no noticeable effect on the performance. saw no reason to remove it though
- Merging euc and man into one 'fast' generalised function is not worth it (40s->60s time)
- refer to 345267b43be9f81abe84dce49259c2827d02ec28 for the merge
"""

# joined types
ctypedef fused scalar_int:
    char
    short
    Py_ssize_t
    int
    long
    long long

ctypedef fused scalar_float:
    float
    double
    long double

cdef inline scalar_int f(scalar_int p, scalar_int posINF) nogil:
    cdef scalar_int out = posINF
    if p == 0:
        out = 0
    return out

cdef inline scalar_int euclidean_dist(scalar_int a, scalar_int b, scalar_int c) nogil:
    cdef scalar_int out = (a-b)**2+c
    return out

cdef inline double euclidean_meet(scalar_int a, scalar_int b, scalar_int[:] f, scalar_int posINF, scalar_int negINF) nogil:
    cdef double out = (f[a]+a**2-f[b]-b**2)/(2*a-2*b)
    if out != out:
        if a==posINF and b!=posINF:
            out = -INFINITY
        else:
            out = INFINITY
    return out

cdef inline scalar_int manhattan_dist(scalar_int a, scalar_int b, scalar_int c) nogil:
    cdef scalar_int out
    if a>=b:
        out = a-b+c
    else:
        out = b-a+c
    return out

cdef inline double manhattan_dist_double(scalar_int a, double b, scalar_int c) nogil:
    cdef double out
    if a>=b:
        out = <double>a-<double>b+<double>c
    else:
        out = <double>b-<double>a+<double>c
    return out

cdef inline double manhattan_meet(scalar_int a, scalar_int b, scalar_int[:] f, scalar_int posINF, scalar_int negINF) nogil:
    cdef double s
    cdef scalar_int fa = f[a]
    cdef scalar_int fb = f[b]
    s = (a + fa + b - fb) / 2
    if manhattan_dist_double(a,s,fa) == manhattan_dist_double(b,s,fb):
        return s
    s = (a - fa + b + fb) / 2
    if manhattan_dist_double(a,s,fa) == manhattan_dist_double(b,s,fb):
        return s
    if manhattan_dist(a,a,fa) > manhattan_dist(b,a,fb):
        return INFINITY
    return -1

def _generalized_distance_transform_1d_euclidean(scalar_int[:] arr, scalar_int[:] cost_arr,
                                       bint isfirst, double[::1] domains,
                                       scalar_int[::1] centers, scalar_int[::1] out, type typeINF, scalar_int negINF, scalar_int posINF):
    cdef scalar_int length = len(arr)
    cdef scalar_int i, rightmost, current_domain,start
    cdef double intersection
    with nogil:
        if isfirst:
            for i in range(length):
                cost_arr[i] = f(arr[i], posINF)

        start = 0
        while start<length:
            if cost_arr[start] != posINF:
                break
            start+=1
        start = min(length-1,start)

        rightmost = 0
        domains[0] = <double>negINF
        domains[1] = <double>posINF
        centers[0] = start
    
        for i in range(start+1,length):
            intersection = euclidean_meet(i,centers[rightmost],cost_arr, posINF, negINF)
            while intersection <= domains[rightmost] or domains[rightmost]==posINF and rightmost>start:
                rightmost-=1
                intersection = euclidean_meet(i,centers[rightmost],cost_arr, posINF, negINF)

            rightmost+=1
            centers[rightmost]=i
            domains[rightmost]=intersection
        domains[rightmost+1] = posINF

        current_domain = 0

        for i in range(length):
            while domains[current_domain+1]<i:
                current_domain += 1
            out[i] = euclidean_dist(i,centers[current_domain],cost_arr[<scalar_int>centers[current_domain]])
    return out

def _generalized_distance_transform_1d_manhattan(scalar_int[:] arr, scalar_int[:] cost_arr,
                                       bint isfirst, double[::1] domains,
                                       scalar_int[::1] centers, scalar_int[::1] out, type typeINF, scalar_int negINF, scalar_int posINF):
    cdef scalar_int length = len(arr)
    cdef scalar_int i, rightmost, current_domain, start
    cdef double intersection
    with nogil:
        if isfirst:
            for i in range(length):
                cost_arr[i] = f(arr[i], posINF)

        start = 0
        while start<length:
            if cost_arr[start] != posINF:
                break
            start+=1
        start = min(length-1,start)

        rightmost = 0
        domains[0] = negINF
        domains[1] = posINF
        centers[0] = start

        for i in range(start+1,length):
            intersection = manhattan_meet(i,<scalar_int>centers[rightmost],cost_arr, posINF, negINF)
            while intersection <= domains[rightmost] or domains[rightmost]==posINF and rightmost>start:
                rightmost-=1
                intersection = manhattan_meet(i,<scalar_int>centers[rightmost],cost_arr, posINF, negINF)

            rightmost+=1
            centers[rightmost]=i
            domains[rightmost]=intersection
        domains[rightmost+1] = posINF

        current_domain = 0

        for i in range(length):
            while domains[current_domain+1]<i:
                current_domain += 1
            out[i] = manhattan_dist(i,centers[current_domain],cost_arr[centers[current_domain]])
    return out

def _generalized_distance_transform_1d_slow(double[:] arr,double[:] cost_arr,
                                           cost_func, dist_func, dist_meet,
                                       bint isfirst, double[::1] domains,
                                       scalar_int[::1] centers, double[::1] out):
    cdef scalar_int length = len(arr)
    cdef scalar_int i, rightmost, current_domain, start
    cdef double intersection

    if isfirst:
        for i in range(length):
            cost_arr[i] = cost_func(arr[i])

    start = 0
    while start<length:
        if cost_arr[start] != INFINITY:
            break
        start+=1
    start = min(length-1,start)

    rightmost = 0
    domains[0] = -INFINITY
    domains[1] = INFINITY
    centers[0] = start

    for i in range(start+1,length):
        intersection = dist_meet(i,centers[rightmost],cost_arr, INFINITY, -INFINITY)
        while intersection <= domains[rightmost] or domains[rightmost]==INFINITY and rightmost>start:
            rightmost-=1
            intersection = dist_meet(i,centers[rightmost],cost_arr, INFINITY, -INFINITY)

        rightmost+=1
        centers[rightmost]=i
        domains[rightmost]=intersection
    domains[rightmost+1] = INFINITY

    current_domain = 0

    for i in range(length):
        while domains[current_domain+1]<i:
            current_domain += 1
        out[i] = dist_func(i,centers[current_domain],cost_arr[centers[current_domain]])
    return out