#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np

cimport numpy as cnp
cnp.import_array()


cdef inline float tri_max(float a, float b, float c):
    cdef float MAX

    if a > b:
        MAX = a
    else:
        MAX = b

    if MAX > c:
        return MAX
    else:
        return c


def histograms(cnp.ndarray[cnp.uint8_t, ndim=3] img, int nbins):
    '''Calculate the channel histograms of the current image.

    Parameters
    ----------
    img : ndarray, uint8, ndim=3
        The image to calculate the histogram.
    nbins : int
        The number of bins.

    Returns
    -------
    out : (rcounts, gcounts, bcounts, vcounts)
        The binned histograms of the RGB channels and grayscale intensity.

    This is a NAIVE histogram routine, meant primarily for fast display.

    '''
    cdef int width = img.shape[1]
    cdef int height = img.shape[0]
    cdef cnp.ndarray[cnp.int32_t, ndim=1] r, g, b, v

    r = np.zeros((nbins,), dtype=np.int32)
    g = np.zeros((nbins,), dtype=np.int32)
    b = np.zeros((nbins,), dtype=np.int32)
    v = np.zeros((nbins,), dtype=np.int32)

    cdef int i, j, k, rbin, gbin, bbin, vbin
    cdef float bin_width = 255./ nbins
    cdef float R, G, B, V

    for i in range(height):
        for j in range(width):
            R = <float>img[i, j, 0]
            G = <float>img[i, j, 1]
            B = <float>img[i, j, 2]
            V = tri_max(R, G, B)

            rbin = <int>(R / bin_width)
            gbin = <int>(G / bin_width)
            bbin = <int>(B / bin_width)
            vbin = <int>(V / bin_width)

            # fully open last bin
            if R == 255:
                rbin -= 1
            if G == 255:
                gbin -= 1
            if B == 255:
                bbin -= 1
            if V == 255:
                vbin -= 1

            r[rbin] += 1
            g[gbin] += 1
            b[bbin] += 1
            v[vbin] += 1

    return (r, g, b, v)
