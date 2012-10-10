""" to compile this use:
>>> python setup.py build_ext --inplace

to generate html report use:
>>> cython -a crank.pxd

"""

#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np

# import main loop
from _core8 cimport _core8

# -----------------------------------------------------------------
# kernels uint8
# -----------------------------------------------------------------

cdef inline np.uint8_t kernel_autolevel(int* histo, float pop, np.uint8_t g):
    cdef int i,imin,imax,delta

    if pop:
        for i in range(255,-1,-1):
            if histo[i]:
                imax = i
                break
        for i in range(256):
            if histo[i]:
                imin = i
                break
    delta = imax-imin
    if delta>0:
        return <np.uint8_t>(255.*(g-imin)/delta)
    else:
        return <np.uint8_t>(imax-imin)

cdef inline np.uint8_t kernel_bottomhat(int* histo, float pop, np.uint8_t g):
    cdef int i

    for i in range(256):
        if histo[i]:
            break

    return <np.uint8_t>(g-i)


cdef inline np.uint8_t kernel_equalize(int* histo, float pop, np.uint8_t g):
    cdef int i
    cdef float sum = 0.

    if pop:
        for i in range(256):
            sum += histo[i]
            if i>=g:
                break

        return <np.uint8_t>((255*sum)/pop)
    else:
        return <np.uint8_t>(0)

cdef inline np.uint8_t kernel_gradient(int* histo, float pop, np.uint8_t g):
    cdef int i,imin,imax


    if pop:
        for i in range(255,-1,-1):
            if histo[i]:
                imax = i
                break
        for i in range(256):
            if histo[i]:
                imin = i
                break
        return <np.uint8_t>(imax-imin)
    else:
        return <np.uint8_t>(0)

cdef inline np.uint8_t kernel_maximum(int* histo, float pop, np.uint8_t g):
    cdef int i

    if pop:
        for i in range(255,-1,-1):
            if histo[i]:
                return <np.uint8_t>(i)

    return <np.uint8_t>(0)

cdef inline np.uint8_t kernel_mean(int* histo, float pop, np.uint8_t g):
    cdef int i
    cdef float mean = 0.

    if pop:
        for i in range(256):
            mean += histo[i]*i
        return <np.uint8_t>(mean/pop)
    else:
        return <np.uint8_t>(0)

cdef inline np.uint8_t kernel_meansubstraction(int* histo, float pop, np.uint8_t g):
    cdef int i
    cdef float mean = 0.

    if pop:
        for i in range(256):
            mean += histo[i]*i
        return <np.uint8_t>((g-mean/pop)/2.+127)
    else:
        return <np.uint8_t>(0)

cdef inline np.uint8_t kernel_median(int* histo, float pop, np.uint8_t g):
    cdef int i
    cdef float sum = pop/2.0

    if pop:
        for i in range(256):
            if histo[i]:
                sum -= histo[i]
                if sum<0:
                    return <np.uint8_t>(i)

    return <np.uint8_t>(0)

cdef inline np.uint8_t kernel_minimum(int* histo, float pop, np.uint8_t g):
    cdef int i

    if pop:
        for i in range(256):
            if histo[i]:
                return <np.uint8_t>(i)

    return <np.uint8_t>(0)

cdef inline np.uint8_t kernel_modal(int* histo, float pop, np.uint8_t g):
    cdef int hmax=0,imax=0

    if pop:
        for i in range(256):
            if histo[i]>hmax:
                hmax = histo[i]
                imax = i
        return <np.uint8_t>(imax)

    return <np.uint8_t>(0)

cdef inline np.uint8_t kernel_morph_contr_enh(int* histo, float pop, np.uint8_t g):
    cdef int i,imin,imax

    if pop:
        for i in range(255,-1,-1):
            if histo[i]:
                imax = i
                break
        for i in range(256):
            if histo[i]:
                imin = i
                break
        if imax-g < g-imin:
            return <np.uint8_t>(imax)
        else:
            return <np.uint8_t>(imin)
    else:
        return <np.uint8_t>(0)

cdef inline np.uint8_t kernel_pop(int* histo, float pop, np.uint8_t g):
    return <np.uint8_t>(pop)

cdef inline np.uint8_t kernel_threshold(int* histo, float pop, np.uint8_t g):
    cdef int i
    cdef float mean = 0.

    if pop:
        for i in range(256):
            mean += histo[i]*i
        return <np.uint8_t>(g>(mean/pop))
    else:
        return <np.uint8_t>(0)

cdef inline np.uint8_t kernel_tophat(int* histo, float pop, np.uint8_t g):
    cdef int i

    for i in range(255,-1,-1):
        if histo[i]:
            break

    return <np.uint8_t>(i-g)

# -----------------------------------------------------------------
# python wrappers
# -----------------------------------------------------------------
def autolevel(np.ndarray[np.uint8_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint8_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0):
    """bottom hat
    """
    return _core8(kernel_autolevel,image,selem,mask,out,shift_x,shift_y)

def bottomhat(np.ndarray[np.uint8_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint8_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0):
    """bottom hat
    """
    return _core8(kernel_bottomhat,image,selem,mask,out,shift_x,shift_y)

def equalize(np.ndarray[np.uint8_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint8_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0):
    """local egalisation of the gray level
    """
    return _core8(kernel_equalize,image,selem,mask,out,shift_x,shift_y)

def gradient(np.ndarray[np.uint8_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint8_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0):
    """local maximum - local minimum gray level
    """
    return _core8(kernel_gradient,image,selem,mask,out,shift_x,shift_y)

def maximum(np.ndarray[np.uint8_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint8_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0):
    """local maximum gray level
    """
    return _core8(kernel_maximum,image,selem,mask,out,shift_x,shift_y)

def mean(np.ndarray[np.uint8_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint8_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0):
    """average gray level (clipped on uint8)
    """
    return _core8(kernel_mean,image,selem,mask,out,shift_x,shift_y)

def meansubstraction(np.ndarray[np.uint8_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint8_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0):
    """(g - average gray level)/2+127 (clipped on uint8)
    """
    return _core8(kernel_meansubstraction,image,selem,mask,out,shift_x,shift_y)

def median(np.ndarray[np.uint8_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint8_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0):
    """local median
    """
    return _core8(kernel_median,image,selem,mask,out,shift_x,shift_y)

def minimum(np.ndarray[np.uint8_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint8_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0):
    """local minimum gray level
    """
    return _core8(kernel_minimum,image,selem,mask,out,shift_x,shift_y)

def morph_contr_enh(np.ndarray[np.uint8_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint8_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0):
    """morphological contrast enhancement
    """
    return _core8(kernel_morph_contr_enh,image,selem,mask,out,shift_x,shift_y)

def modal(np.ndarray[np.uint8_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint8_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0):
    """local mode
    """
    return _core8(kernel_modal,image,selem,mask,out,shift_x,shift_y)

def pop(np.ndarray[np.uint8_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint8_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0):
    """returns the number of actual pixels of the structuring element inside the mask
    """
    return _core8(kernel_pop,image,selem,mask,out,shift_x,shift_y)

def threshold(np.ndarray[np.uint8_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint8_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0):
    """returns 255 if gray level higher than local mean, 0 else
    """
    return _core8(kernel_threshold,image,selem,mask,out,shift_x,shift_y)

def tophat(np.ndarray[np.uint8_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint8_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0):
    """top hat
    """
    return _core8(kernel_tophat,image,selem,mask,out,shift_x,shift_y)

