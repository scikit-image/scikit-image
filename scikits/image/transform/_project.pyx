#cython: cdivison=True boundscheck=False

cimport cython
cimport numpy as np

import numpy as np
import cython

np.import_array()

cdef extern from "math.h":
    double floor(double)
    double fmod(double, double)

cdef double get_pixel(np.ndarray image, int r, int c, char mode,
                      double cval=0):
    cdef np.ndarray[dtype=np.double_t, ndim=2] img = image
    cdef int rows = img.shape[0]
    cdef int cols = img.shape[1]

    if mode == 'C':
        if (r < 0) or (r >= cols) or (c < 0) or (c >= cols):
            return cval
        else:
            return img[r, c]
    else:
        return img[coord_map(rows, r, mode),
                   coord_map(cols, c, mode)]

cdef int coord_map(int dim, int coord, char mode):
    dim = dim - 1
    if mode == 'M': # mirror
        if (coord < 0):
            return <int>(-coord % dim)
        elif (coord > dim):
            return <int>(dim  - (coord % dim))
    elif mode == 'W': # wrap
        if (coord < 0):
            return <int>(dim - (-coord % dim))
        elif (coord > dim):
            return <int>(coord % dim)

    return coord

cdef tf(double x, double y, H):
    cdef np.ndarray[np.double_t, ndim=2] M = H
    cdef double xx, yy, zz

    xx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    yy = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    zz = M[2, 0] * x + M[2, 1] * y + M[2, 2]

    xx /= zz
    yy /= zz

    return xx, yy

@cython.boundscheck(False)
def homography(np.ndarray image, np.ndarray H, output_shape=None,
               mode='C', double cval=0):
    """
    Projective transformation (homography).
    
    Perform a projective transformation (homography) of a
    floating point image, using bi-linear interpolation.

    For each pixel, given its homogeneous coordinate :math:`\mathbf{x}
    = [x, y, 1]^T`, its target position is calculated by multiplying
    with the given matrix, :math:`H`, to give :math:`H \mathbf{x}`.
    E.g., to rotate by theta degrees clockwise, the matrix should be

    ::

      [[cos(theta) -sin(theta) 0]
       [sin(theta)  cos(theta) 0]
       [0            0         1]]

    or, to translate x by 10 and y by 20,

    ::

      [[1 0 10]
       [0 1 20]
       [0 0 1 ]].

    Parameters
    ----------
    image : 2-D array
        Input image.
    H : array of shape ``(3, 3)``
        Transformation matrix H that defines the homography.
    output_shape : tuple (rows, cols)
        Shape of the output image generated.
    order : int
        Order of splines used in interpolation.
    mode : {'C', 'M', 'W'}
        How to handle values outside the image borders.
        Constant, Mirror or Wrap.
    cval : string
        Used in conjunction with mode 'C' (constant), the value
        outside the image boundaries.

    """

    cdef np.ndarray[dtype=np.double_t, ndim=2] img = image
    cdef np.ndarray[dtype=np.double_t, ndim=2] M = np.linalg.inv(H)

    if mode not in ('C', 'W', 'M'):
        raise ValueError("Invalid mode specified.  Please use "
                         "C [constant], W [wrap] or M [mirror].")
    cdef char mode_c = ord(mode[0])

    cdef int out_r, out_c, columns, rows
    if output_shape is None:
        out_r = img.shape[0]
        out_c = img.shape[1]
    else:
        out_r = output_shape[0]
        out_c = output_shape[1]

    rows = img.shape[0]
    columns = img.shape[1]

    cdef np.ndarray[dtype=np.double_t, ndim=2] out = \
         np.zeros((out_r, out_c), dtype=np.float64)
    
    cdef int tfr, tfc, r_int, c_int
    cdef double y0, y1, y2, y3
    cdef double r, c, z, t, u

    for tfr in range(out_r):
        for tfc in range(out_c):
            c, r = tf(tfc, tfr, M)
            r_int = <int>floor(r)
            c_int = <int>floor(c)

            t = r - r_int
            u = c - c_int

            y0 = get_pixel(img, r_int, c_int, mode_c)
            y1 = get_pixel(img, r_int + 1, c_int, mode_c)
            y2 = get_pixel(img, r_int + 1, c_int + 1, mode_c)
            y3 = get_pixel(img, r_int, c_int + 1, mode_c)

            out[tfr, tfc] = \
                (1 - t) * (1 - u) * y0 + \
                t * (1 - u) * y1 + \
                t * u * y2 + (1 - t) * u * y3;

    return out
