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
from core16b cimport rank16b

# -----------------------------------------------------------------
# kernels uint16 take extra parameter for defining the bitdepth
# -----------------------------------------------------------------

#cdef inline np.uint16_t kernel_autolevel(int* histo, float pop, np.uint16_t g,int bitdepth,int maxbin, int midbin):
#    cdef int i,imin,imax,delta
#
#    if pop:
#        for i in range(maxbin-1,-1,-1):
#            if histo[i]:
#                imax = i
#                break
#        for i in range(maxbin):
#            if histo[i]:
#                imin = i
#                break
#    delta = imax-imin
#    if delta>0:
#        return <np.uint16_t>(maxbin*1.*(g-imin)/delta)
#    else:
#        return <np.uint16_t>(imax-imin)
#
#cdef inline np.uint16_t kernel_bottomhat(int* histo, float pop, np.uint16_t g,int bitdepth,int maxbin, int midbin):
#    cdef int i
#
#    for i in range(maxbin):
#        if histo[i]:
#            break
#
#    return <np.uint16_t>(g-i)
#
#
#cdef inline np.uint16_t kernel_egalise(int* histo, float pop, np.uint16_t g,int bitdepth,int maxbin, int midbin):
#    cdef int i
#    cdef float sum = 0.
#
#    if pop:
#        for i in range(maxbin):
#            sum += histo[i]
#            if i>=g:
#                break
#
#        return <np.uint16_t>((maxbin*1.*sum)/pop)
#    else:
#        return <np.uint16_t>(0)
#
#cdef inline np.uint16_t kernel_gradient(int* histo, float pop, np.uint16_t g,int bitdepth,int maxbin, int midbin):
#    cdef int i,imin,imax
#
#    if pop:
#        for i in range(maxbin-1,-1,-1):
#            if histo[i]:
#                imax = i
#                break
#        for i in range(maxbin):
#            if histo[i]:
#                imin = i
#                break
#        return <np.uint16_t>(imax-imin)
#    else:
#        return <np.uint16_t>(0)
#
#cdef inline np.uint16_t kernel_maximum(int* histo, float pop, np.uint16_t g,int bitdepth,int maxbin, int midbin):
#    cdef int i
#
#    if pop:
#        for i in range(maxbin-1,-1,-1):
#            if histo[i]:
#                return <np.uint16_t>(i)
#
#    return <np.uint16_t>(0)

cdef inline np.uint16_t kernel_mean(int* histo, float pop, np.uint16_t g,int bitdepth,int maxbin, int midbin, int s0, int s1):
    cdef int i,bilat_pop=0
    cdef float mean = 0.

    if pop:
        for i in range(maxbin):
            if (g>(i-s0)) and (g<(i+s1)):
                bilat_pop += histo[i]
                mean += histo[i]*i
        if bilat_pop:
            return <np.uint16_t>(mean/bilat_pop)
        else:
            return <np.uint16_t>(0)
    else:
        return <np.uint16_t>(0)

#cdef inline np.uint16_t kernel_meansubstraction(int* histo, float pop, np.uint16_t g,int bitdepth,int maxbin, int midbin):
#    cdef int i
#    cdef float mean = 0.
#
#    if pop:
#        for i in range(maxbin):
#            mean += histo[i]*i
#        return <np.uint16_t>((g-mean/pop)/2.+midbin)
#    else:
#        return <np.uint16_t>(0)
#
#cdef inline np.uint16_t kernel_median(int* histo, float pop, np.uint16_t g,int bitdepth,int maxbin, int midbin):
#    cdef int i
#    cdef float sum = pop/2.0
#
#    if pop:
#        for i in range(maxbin):
#            if histo[i]:
#                sum -= histo[i]
#                if sum<0:
#                    return <np.uint16_t>(i)
#
#    return <np.uint16_t>(0)
#
#cdef inline np.uint16_t kernel_minimum(int* histo, float pop, np.uint16_t g,int bitdepth,int maxbin, int midbin):
#    cdef int i
#
#    if pop:
#        for i in range(maxbin):
#            if histo[i]:
#                return <np.uint16_t>(i)
#
#    return <np.uint16_t>(0)
#
#cdef inline np.uint16_t kernel_modal(int* histo, float pop, np.uint16_t g,int bitdepth,int maxbin, int midbin):
#    cdef int hmax=0,imax=0
#
#    if pop:
#        for i in range(maxbin):
#            if histo[i]>hmax:
#                hmax = histo[i]
#                imax = i
#        return <np.uint16_t>(imax)
#
#    return <np.uint16_t>(0)
#
#cdef inline np.uint16_t kernel_morph_contr_enh(int* histo, float pop, np.uint16_t g,int bitdepth,int maxbin, int midbin):
#    cdef int i,imin,imax
#
#    if pop:
#        for i in range(maxbin-1,-1,-1):
#            if histo[i]:
#                imax = i
#                break
#        for i in range(maxbin):
#            if histo[i]:
#                imin = i
#                break
#        if imax-g < g-imin:
#            return <np.uint16_t>(imax)
#        else:
#            return <np.uint16_t>(imin)
#    else:
#        return <np.uint16_t>(0)
#
cdef inline np.uint16_t kernel_pop(int* histo, float pop, np.uint16_t g,int bitdepth,int maxbin, int midbin, int s0, int s1):
    cdef int i,bilat_pop=0

    if pop:
        for i in range(maxbin):
            if (g>(i-s0)) and (g<(i+s1)):
                bilat_pop += histo[i]
        return <np.uint16_t>(bilat_pop)
    else:
        return <np.uint16_t>(0)

#
#cdef inline np.uint16_t kernel_threshold(int* histo, float pop, np.uint16_t g,int bitdepth,int maxbin, int midbin):
#    cdef int i
#    cdef float mean = 0.
#
#    if pop:
#        for i in range(maxbin):
#            mean += histo[i]*i
#        return <np.uint16_t>(g>(mean/pop))
#    else:
#        return <np.uint16_t>(0)
#
#cdef inline np.uint16_t kernel_tophat(int* histo, float pop, np.uint16_t g,int bitdepth,int maxbin, int midbin):
#    cdef int i
#
#    for i in range(maxbin-1,-1,-1):
#        if histo[i]:
#            break
#
#    return <np.uint16_t>(i-g)

# -----------------------------------------------------------------
# python wrappers
# -----------------------------------------------------------------
#def autolevel(np.ndarray[np.uint16_t, ndim=2] image,
#            np.ndarray[np.uint8_t, ndim=2] selem,
#            np.ndarray[np.uint8_t, ndim=2] mask=None,
#            np.ndarray[np.uint16_t, ndim=2] out=None,
#            char shift_x=0, char shift_y=0, int bitdepth=8, int s0=1, int s1=1):
#    """bottom hat
#    """
#    return rank16b(kernel_autolevel,image,selem,mask,out,shift_x,shift_y,bitdepth,s0,s1)
#
#def bottomhat(np.ndarray[np.uint16_t, ndim=2] image,
#            np.ndarray[np.uint8_t, ndim=2] selem,
#            np.ndarray[np.uint8_t, ndim=2] mask=None,
#            np.ndarray[np.uint16_t, ndim=2] out=None,
#            char shift_x=0, char shift_y=0, int bitdepth=8, int s0=1, int s1=1):
#    """bottom hat
#    """
#    return rank16b(kernel_bottomhat,image,selem,mask,out,shift_x,shift_y,bitdepth,s0,s1)
#
#def egalise(np.ndarray[np.uint16_t, ndim=2] image,
#            np.ndarray[np.uint8_t, ndim=2] selem,
#            np.ndarray[np.uint8_t, ndim=2] mask=None,
#            np.ndarray[np.uint16_t, ndim=2] out=None,
#            char shift_x=0, char shift_y=0, int bitdepth=8, int s0=1, int s1=1):
#    """local egalisation of the gray level
#    """
#    return rank16b(kernel_egalise,image,selem,mask,out,shift_x,shift_y,bitdepth,s0,s1)
#
#def gradient(np.ndarray[np.uint16_t, ndim=2] image,
#            np.ndarray[np.uint8_t, ndim=2] selem,
#            np.ndarray[np.uint8_t, ndim=2] mask=None,
#            np.ndarray[np.uint16_t, ndim=2] out=None,
#            char shift_x=0, char shift_y=0, int bitdepth=8, int s0=1, int s1=1):
#    """local maximum - local minimum gray level
#    """
#    return rank16b(kernel_gradient,image,selem,mask,out,shift_x,shift_y,bitdepth,s0,s1)
#
#def maximum(np.ndarray[np.uint16_t, ndim=2] image,
#            np.ndarray[np.uint8_t, ndim=2] selem,
#            np.ndarray[np.uint8_t, ndim=2] mask=None,
#            np.ndarray[np.uint16_t, ndim=2] out=None,
#            char shift_x=0, char shift_y=0, int bitdepth=8, int s0=1, int s1=1):
#    """local maximum gray level
#    """
#    return rank16b(kernel_maximum,image,selem,mask,out,shift_x,shift_y,bitdepth,s0,s1)

def mean(np.ndarray[np.uint16_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint16_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0, int bitdepth=8, int s0=1, int s1=1):
    """average gray level (clipped on uint8)
    """
    return rank16b(kernel_mean,image,selem,mask,out,shift_x,shift_y,bitdepth,s0,s1)

#def meansubstraction(np.ndarray[np.uint16_t, ndim=2] image,
#            np.ndarray[np.uint8_t, ndim=2] selem,
#            np.ndarray[np.uint8_t, ndim=2] mask=None,
#            np.ndarray[np.uint16_t, ndim=2] out=None,
#            char shift_x=0, char shift_y=0, int bitdepth=8, int s0=1, int s1=1):
#    """(g - average gray level)/2+midbin (clipped on uint8)
#    """
#    return rank16b(kernel_meansubstraction,image,selem,mask,out,shift_x,shift_y,bitdepth,s0,s1)
#
#def median(np.ndarray[np.uint16_t, ndim=2] image,
#            np.ndarray[np.uint8_t, ndim=2] selem,
#            np.ndarray[np.uint8_t, ndim=2] mask=None,
#            np.ndarray[np.uint16_t, ndim=2] out=None,
#            char shift_x=0, char shift_y=0, int bitdepth=8, int s0=1, int s1=1):
#    """local median
#    """
#    return rank16b(kernel_median,image,selem,mask,out,shift_x,shift_y,bitdepth,s0,s1)
#
#def minimum(np.ndarray[np.uint16_t, ndim=2] image,
#            np.ndarray[np.uint8_t, ndim=2] selem,
#            np.ndarray[np.uint8_t, ndim=2] mask=None,
#            np.ndarray[np.uint16_t, ndim=2] out=None,
#            char shift_x=0, char shift_y=0, int bitdepth=8, int s0=1, int s1=1):
#    """local minimum gray level
#    """
#    return rank16b(kernel_minimum,image,selem,mask,out,shift_x,shift_y,bitdepth,s0,s1)
#
#def morph_contr_enh(np.ndarray[np.uint16_t, ndim=2] image,
#            np.ndarray[np.uint8_t, ndim=2] selem,
#            np.ndarray[np.uint8_t, ndim=2] mask=None,
#            np.ndarray[np.uint16_t, ndim=2] out=None,
#            char shift_x=0, char shift_y=0, int bitdepth=8, int s0=1, int s1=1):
#    """morphological contrast enhancement
#    """
#    return rank16b(kernel_morph_contr_enh,image,selem,mask,out,shift_x,shift_y,bitdepth,s0,s1)
#
#def modal(np.ndarray[np.uint16_t, ndim=2] image,
#            np.ndarray[np.uint8_t, ndim=2] selem,
#            np.ndarray[np.uint8_t, ndim=2] mask=None,
#            np.ndarray[np.uint16_t, ndim=2] out=None,
#            char shift_x=0, char shift_y=0, int bitdepth=8, int s0=1, int s1=1):
#    """local mode
#    """
#    return rank16b(kernel_modal,image,selem,mask,out,shift_x,shift_y,bitdepth,s0,s1)
#
def pop(np.ndarray[np.uint16_t, ndim=2] image,
            np.ndarray[np.uint8_t, ndim=2] selem,
            np.ndarray[np.uint8_t, ndim=2] mask=None,
            np.ndarray[np.uint16_t, ndim=2] out=None,
            char shift_x=0, char shift_y=0, int bitdepth=8, int s0=1, int s1=1):
    """returns the number of actual pixels of the structuring element inside the mask
    """
    return rank16b(kernel_pop,image,selem,mask,out,shift_x,shift_y,bitdepth,s0,s1)

#def threshold(np.ndarray[np.uint16_t, ndim=2] image,
#            np.ndarray[np.uint8_t, ndim=2] selem,
#            np.ndarray[np.uint8_t, ndim=2] mask=None,
#            np.ndarray[np.uint16_t, ndim=2] out=None,
#            char shift_x=0, char shift_y=0, int bitdepth=8, int s0=1, int s1=1):
#    """returns maxbin-1 if gray level higher than local mean, 0 else
#    """
#    return rank16b(kernel_threshold,image,selem,mask,out,shift_x,shift_y,bitdepth,s0,s1)
#
#def tophat(np.ndarray[np.uint16_t, ndim=2] image,
#            np.ndarray[np.uint8_t, ndim=2] selem,
#            np.ndarray[np.uint8_t, ndim=2] mask=None,
#            np.ndarray[np.uint16_t, ndim=2] out=None,
#            char shift_x=0, char shift_y=0, int bitdepth=8, int s0=1, int s1=1):
#    """top hat
#    """
#    return rank16b(kernel_tophat,image,selem,mask,out,shift_x,shift_y,bitdepth,s0,s1)
