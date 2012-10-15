""" to compile this use:
>>> python setup.py build_ext --inplace

to generate html report use:
>>> cython -a crank16.pxd

"""

#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np

# import main loop
from _core16 cimport _core16

# -----------------------------------------------------------------
# kernels uint16 take extra parameter for defining the bitdepth
# -----------------------------------------------------------------

cdef inline np.uint16_t kernel_autolevel(Py_ssize_t* histo, float pop, np.uint16_t g, Py_ssize_t bitdepth, Py_ssize_t maxbin, Py_ssize_t midbin):
    cdef Py_ssize_t i,imin,imax,delta

    if pop:
        for i in range(maxbin-1,-1,-1):
            if histo[i]:
                imax = i
                break
        for i in range(maxbin):
            if histo[i]:
                imin = i
                break
    delta = imax-imin
    if delta>0:
        return <np.uint16_t>(1.*(maxbin-1)*(g-imin)/delta)
    else:
        return <np.uint16_t>(imax-imin)

cdef inline np.uint16_t kernel_bottomhat(Py_ssize_t* histo, float pop, np.uint16_t g, Py_ssize_t bitdepth, Py_ssize_t maxbin, Py_ssize_t midbin):
    cdef Py_ssize_t i

    for i in range(maxbin):
        if histo[i]:
            break

    return <np.uint16_t>(g-i)


cdef inline np.uint16_t kernel_equalize(Py_ssize_t* histo, float pop, np.uint16_t g, Py_ssize_t bitdepth, Py_ssize_t maxbin, Py_ssize_t midbin):
    cdef Py_ssize_t i
    cdef float sum = 0.

    if pop:
        for i in range(maxbin):
            sum += histo[i]
            if i>=g:
                break

        return <np.uint16_t>(((maxbin-1)*sum)/pop)
    else:
        return <np.uint16_t>(0)

cdef inline np.uint16_t kernel_gradient(Py_ssize_t* histo, float pop, np.uint16_t g, Py_ssize_t bitdepth, Py_ssize_t maxbin, Py_ssize_t midbin):
    cdef Py_ssize_t i,imin,imax

    if pop:
        for i in range(maxbin-1,-1,-1):
            if histo[i]:
                imax = i
                break
        for i in range(maxbin):
            if histo[i]:
                imin = i
                break
        return <np.uint16_t>(imax-imin)
    else:
        return <np.uint16_t>(0)

cdef inline np.uint16_t kernel_maximum(Py_ssize_t* histo, float pop, np.uint16_t g, Py_ssize_t bitdepth, Py_ssize_t maxbin, Py_ssize_t midbin):
    cdef Py_ssize_t i

    if pop:
        for i in range(maxbin-1,-1,-1):
            if histo[i]:
                return <np.uint16_t>(i)

    return <np.uint16_t>(0)

cdef inline np.uint16_t kernel_mean(Py_ssize_t* histo, float pop, np.uint16_t g, Py_ssize_t bitdepth, Py_ssize_t maxbin, Py_ssize_t midbin):
    cdef Py_ssize_t i
    cdef float mean = 0.

    if pop:
        for i in range(maxbin):
            mean += histo[i]*i
        return <np.uint16_t>(mean/pop)
    else:
        return <np.uint16_t>(0)

cdef inline np.uint16_t kernel_meansubstraction(Py_ssize_t* histo, float pop, np.uint16_t g, Py_ssize_t bitdepth, Py_ssize_t maxbin, Py_ssize_t midbin):
    cdef Py_ssize_t i
    cdef float mean = 0.

    if pop:
        for i in range(maxbin):
            mean += histo[i]*i
        return <np.uint16_t>((g-mean/pop)/2.+(midbin-1))
    else:
        return <np.uint16_t>(0)

cdef inline np.uint16_t kernel_median(Py_ssize_t* histo, float pop, np.uint16_t g, Py_ssize_t bitdepth, Py_ssize_t maxbin, Py_ssize_t midbin):
    cdef Py_ssize_t i
    cdef float sum = pop/2.0

    if pop:
        for i in range(maxbin):
            if histo[i]:
                sum -= histo[i]
                if sum<0:
                    return <np.uint16_t>(i)

    return <np.uint16_t>(0)

cdef inline np.uint16_t kernel_minimum(Py_ssize_t* histo, float pop, np.uint16_t g, Py_ssize_t bitdepth, Py_ssize_t maxbin, Py_ssize_t midbin):
    cdef Py_ssize_t i

    if pop:
        for i in range(maxbin):
            if histo[i]:
                return <np.uint16_t>(i)

    return <np.uint16_t>(0)

cdef inline np.uint16_t kernel_modal(Py_ssize_t* histo, float pop, np.uint16_t g, Py_ssize_t bitdepth, Py_ssize_t maxbin, Py_ssize_t midbin):
    cdef Py_ssize_t hmax=0,imax=0

    if pop:
        for i in range(maxbin):
            if histo[i]>hmax:
                hmax = histo[i]
                imax = i
        return <np.uint16_t>(imax)

    return <np.uint16_t>(0)

cdef inline np.uint16_t kernel_morph_contr_enh(Py_ssize_t* histo, float pop, np.uint16_t g, Py_ssize_t bitdepth, Py_ssize_t maxbin, Py_ssize_t midbin):
    cdef Py_ssize_t i,imin,imax

    if pop:
        for i in range(maxbin-1,-1,-1):
            if histo[i]:
                imax = i
                break
        for i in range(maxbin):
            if histo[i]:
                imin = i
                break
        if imax-g < g-imin:
            return <np.uint16_t>(imax)
        else:
            return <np.uint16_t>(imin)
    else:
        return <np.uint16_t>(0)

cdef inline np.uint16_t kernel_pop(Py_ssize_t* histo, float pop, np.uint16_t g, Py_ssize_t bitdepth, Py_ssize_t maxbin, Py_ssize_t midbin):
    return <np.uint16_t>(pop)

cdef inline np.uint16_t kernel_threshold(Py_ssize_t* histo, float pop, np.uint16_t g, Py_ssize_t bitdepth, Py_ssize_t maxbin, Py_ssize_t midbin):
    cdef Py_ssize_t i
    cdef float mean = 0.

    if pop:
        for i in range(maxbin):
            mean += histo[i]*i
        return <np.uint16_t>(g>(mean/pop))
    else:
        return <np.uint16_t>(0)

cdef inline np.uint16_t kernel_tophat(Py_ssize_t* histo, float pop, np.uint16_t g, Py_ssize_t bitdepth, Py_ssize_t maxbin, Py_ssize_t midbin):
    cdef Py_ssize_t i

    for i in range(maxbin-1,-1,-1):
        if histo[i]:
            break

    return <np.uint16_t>(i-g)

# -----------------------------------------------------------------
# python wrappers
# -----------------------------------------------------------------
def autolevel(np.ndarray[np.uint16_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint16_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    """bottom hat
    """
    return _core16(kernel_autolevel,image,selem,mask,out,shift_x,shift_y,bitdepth)

def bottomhat(np.ndarray[np.uint16_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint16_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    """bottom hat
    """
    return _core16(kernel_bottomhat,image,selem,mask,out,shift_x,shift_y,bitdepth)

def equalize(np.ndarray[np.uint16_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint16_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    """local egalisation of the gray level
    """
    return _core16(kernel_equalize,image,selem,mask,out,shift_x,shift_y,bitdepth)

def gradient(np.ndarray[np.uint16_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint16_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    """local maximum - local minimum gray level
    """
    return _core16(kernel_gradient,image,selem,mask,out,shift_x,shift_y,bitdepth)

def maximum(np.ndarray[np.uint16_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint16_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    """local maximum gray level
    """
    return _core16(kernel_maximum,image,selem,mask,out,shift_x,shift_y,bitdepth)

def mean(np.ndarray[np.uint16_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint16_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    """average gray level (clipped on uint8)
    """
    return _core16(kernel_mean,image,selem,mask,out,shift_x,shift_y,bitdepth)

def meansubstraction(np.ndarray[np.uint16_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint16_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    """(g - average gray level)/2+midbin (clipped on uint8)
    """
    return _core16(kernel_meansubstraction,image,selem,mask,out,shift_x,shift_y,bitdepth)

def median(np.ndarray[np.uint16_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint16_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    """local median
    """
    return _core16(kernel_median,image,selem,mask,out,shift_x,shift_y,bitdepth)

def minimum(np.ndarray[np.uint16_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint16_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    """local minimum gray level
    """
    return _core16(kernel_minimum,image,selem,mask,out,shift_x,shift_y,bitdepth)

def morph_contr_enh(np.ndarray[np.uint16_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint16_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    """morphological contrast enhancement
    """
    return _core16(kernel_morph_contr_enh,image,selem,mask,out,shift_x,shift_y,bitdepth)

def modal(np.ndarray[np.uint16_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint16_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    """local mode
    """
    return _core16(kernel_modal,image,selem,mask,out,shift_x,shift_y,bitdepth)

def pop(np.ndarray[np.uint16_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint16_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    """returns the number of actual pixels of the structuring element inside the mask
    """
    return _core16(kernel_pop,image,selem,mask,out,shift_x,shift_y,bitdepth)

def threshold(np.ndarray[np.uint16_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint16_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    """returns maxbin-1 if gray level higher than local mean, 0 else
    """
    return _core16(kernel_threshold,image,selem,mask,out,shift_x,shift_y,bitdepth)

def tophat(np.ndarray[np.uint16_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint16_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0, Py_ssize_t bitdepth=8):
    """top hat
    """
    return _core16(kernel_tophat,image,selem,mask,out,shift_x,shift_y,bitdepth)
