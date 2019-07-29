#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
from numpy cimport ndarray
from numpy.math cimport INFINITY


cdef double f(double p) nogil:
    cdef double out = INFINITY
    if p == 0:
        out = 0
    return out

cdef double euclidean_dist(Py_ssize_t a, Py_ssize_t b, double c) nogil:
    cdef double out = <double>(a-b)**2+c
    return out

cdef double euclidean_meet(Py_ssize_t a, Py_ssize_t b, double[:] f) nogil:
    cdef double out = (f[a]+a**2-f[b]-b**2)/(2*a-2*b)
    if out != out:
        if a==INFINITY and b!=INFINITY:
            out = -INFINITY
        else:
            out = INFINITY
    return out

cdef double manhattan_dist(Py_ssize_t a, double b, double c) nogil:
    cdef double out
    if a>=b:
        out = a-b+c
    else:
        out = b-a+c
    return out

cdef double manhattan_meet(Py_ssize_t a, Py_ssize_t b, double[:] f) nogil:
    cdef double s
    cdef double fa = f[a]
    cdef double fb = f[b]
    s = (a + fa + b - fb) / 2
    if manhattan_dist(a,s,fa) == manhattan_dist(b,s,fb):
        return s
    s = (a - fa + b + fb) / 2
    if manhattan_dist(a,s,fa) == manhattan_dist(b,s,fb):
        return s
    if manhattan_dist(a,a,fa) > manhattan_dist(b,a,fb):
        return INFINITY
    return -1

def _generalized_distance_transform_1d_euclidean(double[:] arr, double[:] cost_arr,
                                       bint isfirst, double[::1] domains,
                                       Py_ssize_t[::1] centers, double[::1] out):

    cdef Py_ssize_t length = len(arr)
    cdef Py_ssize_t i, rightmost, current_domain,start
    cdef double intersection
    with nogil:
        if isfirst:
            for i in range(length):
                cost_arr[i] = f(arr[i])

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
            intersection = euclidean_meet(i,centers[rightmost],cost_arr)
            while intersection <= domains[rightmost] or domains[rightmost]==INFINITY and rightmost>start:
                rightmost-=1
                intersection = euclidean_meet(i,centers[rightmost],cost_arr)

            rightmost+=1
            centers[rightmost]=i
            domains[rightmost]=intersection
        domains[rightmost+1] = INFINITY

        current_domain = 0

        for i in range(length):
            while domains[current_domain+1]<i:
                current_domain += 1
            out[i] = euclidean_dist(i,centers[current_domain],cost_arr[<Py_ssize_t>centers[current_domain]])
    return out

def _generalized_distance_transform_1d_manhattan(double[:] arr, double[:] cost_arr,
                                       bint isfirst, double[::1] domains,
                                       Py_ssize_t[::1] centers, double[::1] out):

    cdef Py_ssize_t length = len(arr)
    cdef Py_ssize_t i, rightmost, current_domain, start
    cdef double intersection
    with nogil:
        if isfirst:
            for i in range(length):
                cost_arr[i] = f(arr[i])

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
            intersection = manhattan_meet(i,<Py_ssize_t>centers[rightmost],cost_arr)
            while intersection <= domains[rightmost] or domains[rightmost]==INFINITY and rightmost>start:
                rightmost-=1
                intersection = manhattan_meet(i,<Py_ssize_t>centers[rightmost],cost_arr)

            rightmost+=1
            centers[rightmost]=i
            domains[rightmost]=intersection
        domains[rightmost+1] = INFINITY

        current_domain = 0

        for i in range(length):
            while domains[current_domain+1]<i:
                current_domain += 1
            manhattan_dist(i,centers[current_domain],cost_arr[centers[current_domain]])
            out[i] = manhattan_dist(i,centers[current_domain],cost_arr[centers[current_domain]])
    return out


def _generalized_distance_transform_1d_slow(double[:] arr,double[:] cost_arr,
                                       cost_func, dist_func, dist_meet,
                                       bint isfirst, double[::1] domains,
                                       Py_ssize_t[::1] centers, double[::1] out):
    cdef Py_ssize_t length = len(arr)
    cdef Py_ssize_t i, rightmost, current_domain, start
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
        intersection = dist_meet(i,centers[rightmost],cost_arr)
        while intersection <= domains[rightmost] or domains[rightmost]==INFINITY and rightmost>start:
            rightmost-=1
            intersection = dist_meet(i,centers[rightmost],cost_arr)

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