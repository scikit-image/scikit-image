#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from skimage.filter.rank._core8 cimport _core8, uint8_max, uint8_min


# -----------------------------------------------------------------
# kernels uint8 (SOFT version using percentiles)
# -----------------------------------------------------------------


ctypedef cnp.uint8_t dtype_t


cdef inline dtype_t kernel_autolevel(Py_ssize_t * histo, float pop,
                                     dtype_t g, float p0, float p1,
                                     Py_ssize_t s0, Py_ssize_t s1):
    cdef int i, imin, imax, sum, delta

    if pop:
        sum = 0
        p1 = 1.0 - p1
        imin = 0
        imax = 255

        for i in range(256):
            sum += histo[i]
            if sum > (p0 * pop):
                imin = i
                break
        sum = 0
        for i in range(255, -1, -1):
            sum += histo[i]
            if sum > (p1 * pop):
                imax = i
                break
        delta = imax - imin
        if delta > 0:
            return <dtype_t>(255 * (uint8_min(uint8_max(imin, g), imax)
                                          - imin) / delta)
        else:
            return <dtype_t>(imax - imin)
    else:
        return <dtype_t>(128)


cdef inline dtype_t kernel_gradient(Py_ssize_t * histo, float pop,
                                    dtype_t g, float p0, float p1,
                                    Py_ssize_t s0, Py_ssize_t s1):
    cdef int i, imin, imax, sum, delta

    if pop:
        sum = 0
        p1 = 1.0 - p1
        for i in range(256):
            sum += histo[i]
            if sum >= p0 * pop:
                imin = i
                break
        sum = 0
        for i in range(255, -1, -1):
            sum += histo[i]
            if sum >= p1 * pop:
                imax = i
                break

        return <dtype_t>(imax - imin)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_mean(Py_ssize_t * histo, float pop,
                                dtype_t g, float p0, float p1,
                                Py_ssize_t s0, Py_ssize_t s1):
    cdef int i, sum, mean, n

    if pop:
        sum = 0
        mean = 0
        n = 0
        for i in range(256):
            sum += histo[i]
            if (sum >= p0 * pop) and (sum <= p1 * pop):
                n += histo[i]
                mean += histo[i] * i
        if n > 0:
            return <dtype_t>(1.0 * mean / n)
        else:
            return <dtype_t>(0)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_mean_substraction(Py_ssize_t * histo,
                                             float pop,
                                             dtype_t g,
                                             float p0, float p1,
                                             Py_ssize_t s0, Py_ssize_t s1):
    cdef int i, sum, mean, n

    if pop:
        sum = 0
        mean = 0
        n = 0
        for i in range(256):
            sum += histo[i]
            if (sum >= p0 * pop) and (sum <= p1 * pop):
                n += histo[i]
                mean += histo[i] * i
        if n > 0:
            return <dtype_t>((g - (mean / n)) * .5 + 127)
        else:
            return <dtype_t>(0)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_morph_contr_enh(Py_ssize_t * histo,
                                           float pop,
                                           dtype_t g, float p0, float p1,
                                           Py_ssize_t s0, Py_ssize_t s1):
    cdef int i, imin, imax, sum, delta

    if pop:
        sum = 0
        p1 = 1.0 - p1
        for i in range(256):
            sum += histo[i]
            if sum >= p0 * pop:
                imin = i
                break
        sum = 0
        for i in range(255, -1, -1):
            sum += histo[i]
            if sum >= p1 * pop:
                imax = i
                break
        if g > imax:
            return <dtype_t>imax
        if g < imin:
            return <dtype_t>imin
        if imax - g < g - imin:
            return <dtype_t>imax
        else:
            return <dtype_t>imin
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_percentile(Py_ssize_t * histo, float pop,
                                      dtype_t g, float p0, float p1,
                                      Py_ssize_t s0, Py_ssize_t s1):
    cdef int i
    cdef float sum = 0.

    if pop:
        if p0==0:
            for i in range(256):
                sum += histo[i]
                if sum > (p0 * pop):
                    break
            return <dtype_t>(i)
        else:
            for i in range(256):
                sum += histo[i]
                if sum >= (p0 * pop):
                    break
            return <dtype_t>(i)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_pop(Py_ssize_t * histo, float pop,
                               dtype_t g, float p0, float p1,
                               Py_ssize_t s0, Py_ssize_t s1):
    cdef int i, sum, n

    if pop:
        sum = 0
        n = 0
        for i in range(256):
            sum += histo[i]
            if (sum >= p0 * pop) and (sum <= p1 * pop):
                n += histo[i]
        return <dtype_t>(n)
    else:
        return <dtype_t>(0)


cdef inline dtype_t kernel_threshold(Py_ssize_t * histo, float pop,
                                     dtype_t g, float p0, float p1,
                                     Py_ssize_t s0, Py_ssize_t s1):
    cdef int i
    cdef float sum = 0.

    if pop:
        for i in range(256):
            sum += histo[i]
            if sum >= p0 * pop:
                break

        return <dtype_t>(255 * (g >= i))
    else:
        return <dtype_t>(0)


# -----------------------------------------------------------------
# python wrappers
# -----------------------------------------------------------------


def autolevel(cnp.ndarray[dtype_t, ndim=2] image,
              cnp.ndarray[dtype_t, ndim=2] selem,
              cnp.ndarray[dtype_t, ndim=2] mask=None,
              cnp.ndarray[dtype_t, ndim=2] out=None,
              char shift_x=0, char shift_y=0, float p0=0., float p1=0.):
    """autolevel
    """
    _core8(kernel_autolevel, image, selem, mask, out, shift_x, shift_y, p0, p1,
           <Py_ssize_t>0, <Py_ssize_t>0)


def gradient(cnp.ndarray[dtype_t, ndim=2] image,
             cnp.ndarray[dtype_t, ndim=2] selem,
             cnp.ndarray[dtype_t, ndim=2] mask=None,
             cnp.ndarray[dtype_t, ndim=2] out=None,
             char shift_x=0, char shift_y=0, float p0=0., float p1=0.):
    """return p0,p1 percentile gradient
    """
    _core8(kernel_gradient, image, selem, mask, out, shift_x, shift_y, p0, p1,
           <Py_ssize_t>0, <Py_ssize_t>0)


def mean(cnp.ndarray[dtype_t, ndim=2] image,
         cnp.ndarray[dtype_t, ndim=2] selem,
         cnp.ndarray[dtype_t, ndim=2] mask=None,
         cnp.ndarray[dtype_t, ndim=2] out=None,
         char shift_x=0, char shift_y=0, float p0=0., float p1=0.):
    """return mean between [p0 and p1] percentiles
    """
    _core8(kernel_mean, image, selem, mask, out, shift_x, shift_y, p0, p1,
           <Py_ssize_t>0, <Py_ssize_t>0)


def mean_substraction(cnp.ndarray[dtype_t, ndim=2] image,
                      cnp.ndarray[dtype_t, ndim=2] selem,
                      cnp.ndarray[dtype_t, ndim=2] mask=None,
                      cnp.ndarray[dtype_t, ndim=2] out=None,
                      char shift_x=0, char shift_y=0, float p0=0., float p1=0.):
    """return original - mean between [p0 and p1] percentiles *.5 +127
    """
    _core8(kernel_mean_substraction, image, selem, mask, out, shift_x, shift_y,
           p0, p1, <Py_ssize_t>0, <Py_ssize_t>0)


def morph_contr_enh(cnp.ndarray[dtype_t, ndim=2] image,
                    cnp.ndarray[dtype_t, ndim=2] selem,
                    cnp.ndarray[dtype_t, ndim=2] mask=None,
                    cnp.ndarray[dtype_t, ndim=2] out=None,
                    char shift_x=0, char shift_y=0, float p0=0., float p1=0.):
    """reforce contrast using percentiles
    """
    _core8(kernel_morph_contr_enh, image, selem, mask, out, shift_x, shift_y,
           p0, p1, <Py_ssize_t>0, <Py_ssize_t>0)


def percentile(cnp.ndarray[dtype_t, ndim=2] image,
               cnp.ndarray[dtype_t, ndim=2] selem,
               cnp.ndarray[dtype_t, ndim=2] mask=None,
               cnp.ndarray[dtype_t, ndim=2] out=None,
               char shift_x=0, char shift_y=0, float p0=0., float p1=0.):
    """return p0 percentile
    """
    _core8(kernel_percentile, image, selem, mask, out, shift_x, shift_y,
           p0, 0., <Py_ssize_t>0, <Py_ssize_t>0)


def pop(cnp.ndarray[dtype_t, ndim=2] image,
        cnp.ndarray[dtype_t, ndim=2] selem,
        cnp.ndarray[dtype_t, ndim=2] mask=None,
        cnp.ndarray[dtype_t, ndim=2] out=None,
        char shift_x=0, char shift_y=0, float p0=0., float p1=0.):
    """return nb of pixels between [p0 and p1]
    """
    _core8(kernel_pop, image, selem, mask, out, shift_x, shift_y, p0, p1,
           <Py_ssize_t>0, <Py_ssize_t>0)


def threshold(cnp.ndarray[dtype_t, ndim=2] image,
              cnp.ndarray[dtype_t, ndim=2] selem,
              cnp.ndarray[dtype_t, ndim=2] mask=None,
              cnp.ndarray[dtype_t, ndim=2] out=None,
              char shift_x=0, char shift_y=0, float p0=0., float p1=0.):
    """return 255 if g > percentile p0
    """
    _core8(kernel_threshold, image, selem, mask, out, shift_x, shift_y, p0, 0.,
           <Py_ssize_t>0, <Py_ssize_t>0)
