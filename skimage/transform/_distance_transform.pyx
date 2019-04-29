#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
from numpy cimport ndarray
from numpy.math cimport INFINITY

cdef double f(double p):
    cdef double out = 1.7976931348623157e+308 #largest number a float64 can take
    if p == 0:
        out = 0
    return out

cdef double euclidean_dist(double a, double b, double c):
    cdef double out = (a-b)**2+c
    return out

cdef double euclidean_meet(Py_ssize_t a, Py_ssize_t b, double[:] f):
    cdef double fa = f[a]
    cdef double fb = f[b]
    cdef double top = (fa+a**2-fb-b**2)
    cdef double bottom = (2*a-2*b)
    cdef double out = top/bottom
    return out

cdef manhattan_dist(double a, double b, double c):
    cdef double out = (a-b)**2+c
    return out

cdef manhattan_meet(double a, double b, ndarray[double, ndim=1] f):
    cdef double s
    cdef double fa = f[a]
    cdef double fb = f[b]
    s = (a + fa + b - fb) / 2
    if manhattan_dist(a,s,fa)==manhattan_dist(b,s,fb):
        return s
    s = (a - fa + b + fb) / 2
    if manhattan_dist(a,s,fa)==manhattan_dist(b,s,fb):
        return s
    if manhattan_dist(a,a,fa) > manhattan_dist(b,a,fb):
        return 1.7976931348623157e+308 #largest number a float64 can take
    return -1.7976931348623157e+308 #smallest number a float64 can take

def _generalized_distance_transform_1d_euclidean(double[:] arr, double[:] cost_arr,
                                       bint isfirst, double[::1] domains,
                                       Py_ssize_t[::1] centers, double[::1] out):
    cdef Py_ssize_t length = len(arr)
    cdef Py_ssize_t i, rightmost, current_domain
    cdef double intersection

    if isfirst:
        for i in range(length):
            cost_arr[i] = f(arr[i])

    rightmost = 0
    domains[0] = -INFINITY
    domains[1] = INFINITY
    
    for i in range(1,length):
        intersection = euclidean_meet(i,centers[rightmost],cost_arr)
        while intersection <= domains[rightmost]:
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
        out[i] = euclidean_dist(i,centers[current_domain],cost_arr[centers[current_domain]])
    return out

def _generalized_distance_transform_1d_manhattan(double[:] arr, double[:] cost_arr,
                                       bint isfirst, double[::1] domains,
                                       Py_ssize_t[::1] centers, double[::1] out):
    cdef Py_ssize_t length = len(arr)
    cdef Py_ssize_t i, rightmost, current_domain
    cdef double intersection

    if isfirst:
        for i in range(length):
            cost_arr[i] = f(arr[i])

    rightmost = 0
    domains[0] = -INFINITY
    domains[1] = INFINITY
    
    for i in range(1,length):
        intersection = manhattan_meet(i,centers[rightmost],cost_arr)
        while intersection <= domains[rightmost]:
            rightmost-=1
            intersection = manhattan_meet(i,centers[rightmost],cost_arr)

        rightmost+=1
        centers[rightmost]=i
        domains[rightmost]=intersection
        domains[rightmost+1] = INFINITY

    current_domain = 0

    for i in range(length):
        while domains[current_domain+1]<i:
            current_domain += 1
        out[i] = manhattan_dist(i,centers[current_domain],cost_arr[centers[current_domain]])
    return out


def _generalized_distance_transform_1d_slow(double[:] arr,double[:] cost_arr,
                                       cost_func, dist_func, dist_meet,
                                       bint isfirst, double[::1] domains,
                                       Py_ssize_t[::1] centers, double[::1] out):
    cdef Py_ssize_t length = len(arr)
    cdef Py_ssize_t i, rightmost, current_domain
    cdef double intersection

    if isfirst:
        for i in range(length):
            cost_arr[i] = cost_func(arr[i])

    rightmost = 0
    domains[0] = -INFINITY
    domains[1] = INFINITY
    
    for i in range(1,length):
        intersection = dist_meet(i,centers[rightmost],cost_arr)
        while intersection <= domains[rightmost]:
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