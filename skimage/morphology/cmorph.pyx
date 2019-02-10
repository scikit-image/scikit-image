#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free


def _dilate(np.uint8_t[:, :] image,
            np.uint8_t[:, :] selem,
            np.uint8_t[:, :] out=None,
            signed char shift_x=0, signed char shift_y=0):
    """Return greyscale morphological dilation of an image.

    Morphological dilation sets a pixel at (i,j) to the maximum over all pixels
    in the neighborhood centered at (i,j). Dilation enlarges bright regions
    and shrinks dark regions.

    Parameters
    ----------

    image : ndarray
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        The array to store the result of the morphology. If None, is
        passed, a new array will be allocated.
    shift_x, shift_y : bool
        shift structuring element about center point. This only affects
        eccentric structuring elements (i.e. selem with even numbered sides).

    Returns
    -------
    dilated : uint8 array
        The result of the morphological dilation.
    """

    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]
    cdef Py_ssize_t srows = selem.shape[0]
    cdef Py_ssize_t scols = selem.shape[1]

    cdef Py_ssize_t centre_r = int(selem.shape[0] / 2) - shift_y
    cdef Py_ssize_t centre_c = int(selem.shape[1] / 2) - shift_x

    image = np.ascontiguousarray(image)
    if out is None:
        out = np.zeros((rows, cols), dtype=np.uint8)

    cdef Py_ssize_t r, c, rr, cc, s, value, local_max

    cdef Py_ssize_t selem_num = np.sum(np.asarray(selem) != 0)
    cdef Py_ssize_t* sr = <Py_ssize_t*>malloc(selem_num * sizeof(Py_ssize_t))
    cdef Py_ssize_t* sc = <Py_ssize_t*>malloc(selem_num * sizeof(Py_ssize_t))
    if sr is NULL or sc is NULL:
        free(sr)
        free(sc)
        raise MemoryError()

    s = 0
    for r in range(srows):
        for c in range(scols):
            if selem[r, c] != 0:
                sr[s] = r - centre_r
                sc[s] = c - centre_c
                s += 1

    for r in range(rows):
        for c in range(cols):
            local_max = 0
            for s in range(selem_num):
                rr = r + sr[s]
                cc = c + sc[s]
                if 0 <= rr < rows and 0 <= cc < cols:
                    value = image[rr, cc]
                    if value > local_max:
                        local_max = value

            out[r, c] = local_max

    free(sr)
    free(sc)

    return np.asarray(out)


def _erode(np.uint8_t[:, :] image,
           np.uint8_t[:, :] selem,
           np.uint8_t[:, :] out=None,
           signed char shift_x=0, signed char shift_y=0):
    """Return greyscale morphological erosion of an image.

    Morphological erosion sets a pixel at (i,j) to the minimum over all pixels
    in the neighborhood centered at (i,j). Erosion shrinks bright regions and
    enlarges dark regions.

    Parameters
    ----------
    image : ndarray
        Image array.
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    shift_x, shift_y : bool
        shift structuring element about center point. This only affects
        eccentric structuring elements (i.e. selem with even numbered sides).

    Returns
    -------
    eroded : uint8 array
        The result of the morphological erosion.
    """

    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]
    cdef Py_ssize_t srows = selem.shape[0]
    cdef Py_ssize_t scols = selem.shape[1]

    cdef Py_ssize_t centre_r = int(selem.shape[0] / 2) - shift_y
    cdef Py_ssize_t centre_c = int(selem.shape[1] / 2) - shift_x

    image = np.ascontiguousarray(image)
    if out is None:
        out = np.zeros((rows, cols), dtype=np.uint8)

    cdef int r, c, rr, cc, s, value, local_min

    cdef Py_ssize_t selem_num = np.sum(np.asarray(selem) != 0)
    cdef Py_ssize_t* sr = <Py_ssize_t*>malloc(selem_num * sizeof(Py_ssize_t))
    cdef Py_ssize_t* sc = <Py_ssize_t*>malloc(selem_num * sizeof(Py_ssize_t))
    if sr is NULL or sc is NULL:
        free(sr)
        free(sc)
        raise MemoryError()

    s = 0
    for r in range(srows):
        for c in range(scols):
            if selem[r, c] != 0:
                sr[s] = r - centre_r
                sc[s] = c - centre_c
                s += 1

    for r in range(rows):
        for c in range(cols):
            local_min = 255
            for s in range(selem_num):
                rr = r + sr[s]
                cc = c + sc[s]
                if 0 <= rr < rows and 0 <= cc < cols:
                    value = image[rr, cc]
                    if value < local_min:
                        local_min = value

            out[r, c] = local_min

    free(sr)
    free(sc)

    return np.asarray(out)
