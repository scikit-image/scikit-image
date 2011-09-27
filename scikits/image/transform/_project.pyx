#cython: cdivison=True boundscheck=False

__all__ = ['homography']

cimport cython
cimport numpy as np

import numpy as np
import cython

from cython.operator import dereference

np.import_array()

cdef extern from "math.h":
    double floor(double)
    double fmod(double, double)

cdef double get_pixel(double *image, int rows, int cols,
                      int r, int c, char mode, double cval=0):
    """Get a pixel from the image, taking wrapping mode into consideration.

    Parameters
    ----------
    image : *double
        Input image.
    rows, cols : int
        Dimensions of image.
    r, c : int
        Position at which to get the pixel.
    mode : {'C', 'W', 'M'}
        Wrapping mode.  Constant, Wrap or Mirror.
    cval : double
        Constant value to use for mode constant.
    
    """
    if mode == 'C':
        if (r < 0) or (r > rows - 1) or (c < 0) or (c > cols - 1):
            return cval
        else:
            return image[r * cols + c]
    else:
        return image[coord_map(rows, r, mode) * cols +
                     coord_map(cols, c, mode)]

cdef int coord_map(int dim, int coord, char mode):
    """
    Wrap a coordinate, according to a given dimension and mode.
    
    Parameters
    ----------
    dim : int
        Maximum coordinate.
    coord : int
        Coord provided by user.  May be < 0 or > dim.
    mode : {'W', 'M'}
        Whether to wrap or mirror the coordinate if it
        falls outside [0, dim).
    
    """
    dim = dim - 1
    if mode == 'M': # mirror
        if (coord < 0):
            # How many times times does the coordinate wrap?
            if (<int>(-coord / dim) % 2 != 0):
                return dim - <int>(-coord % dim)
            else:
                return <int>(-coord % dim)
        elif (coord > dim):
            if (<int>(coord / dim) % 2 != 0):
                return <int>(dim - (coord % dim))
            else:
                return <int>(coord % dim)
    elif mode == 'W': # wrap
        if (coord < 0):
            return <int>(dim - (-coord % dim))
        elif (coord > dim):
            return <int>(coord % dim)

    return coord

cdef tf(double x, double y, double* H, double *x_, double *y_):
    """Apply a homography to a coordinate.

    Parameters
    ----------
    x, y : double
        Input coordinate.
    H : (3,3) *double
        Transformation matrix.
    x_, y_ : *double
        Output coordinate.

    """
    cdef double xx, yy, zz

    xx = H[0] * x + H[1] * y + H[2]
    yy = H[3] * x + H[4] * y + H[5]
    zz =  H[6] * x + H[7] * y + H[8]

    xx = xx / zz
    yy = yy / zz

    x_[0] = xx
    y_[0] = yy

@cython.boundscheck(False)
def homography(np.ndarray image, np.ndarray H, output_shape=None,
               mode='constant', double cval=0):
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
    mode : {'constant', 'mirror', 'wrap'}
        How to handle values outside the image borders.
    cval : string
        Used in conjunction with mode 'C' (constant), the value
        outside the image boundaries.

    """

    cdef np.ndarray[dtype=np.double_t, ndim=2, mode="c"] img = \
         np.ascontiguousarray(image, dtype=np.double)
    cdef np.ndarray[dtype=np.double_t, ndim=2, mode="c"] M = \
         np.ascontiguousarray(np.linalg.inv(H))

    if mode not in ('constant', 'wrap', 'mirror'):
        raise ValueError("Invalid mode specified.  Please use "
                         "`constant`, `wrap` or `mirror`.")
    if mode == 'constant':
        mode_c = ord('C')
    elif mode == 'wrap':
        mode_c = ord('W')
    elif mode == 'mirror':
        mode_c = ord('M')

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
         np.zeros((out_r, out_c), dtype=np.double)
    
    cdef int tfr, tfc, r_int, c_int
    cdef double y0, y1, y2, y3
    cdef double r, c, z, t, u

    for tfr in range(out_r):
        for tfc in range(out_c):
            tf(tfc, tfr, <double*>M.data, &c, &r)
            r_int = <int>floor(r)
            c_int = <int>floor(c)

            t = r - r_int
            u = c - c_int

            y0 = get_pixel(<double*>img.data, rows, columns,
                           r_int, c_int, mode_c)
            y1 = get_pixel(<double*>img.data, rows, columns,
                           r_int + 1, c_int, mode_c)
            y2 = get_pixel(<double*>img.data, rows, columns,
                           r_int + 1, c_int + 1, mode_c)
            y3 = get_pixel(<double*>img.data, rows, columns,
                           r_int, c_int + 1, mode_c)

            out[tfr, tfc] = \
                (1 - t) * (1 - u) * y0 + \
                t * (1 - u) * y1 + \
                t * u * y2 + (1 - t) * u * y3;

    return out
