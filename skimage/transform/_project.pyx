#cython: cdivison=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as np
import numpy as np
from cython.operator import dereference
from libc.math cimport ceil, floor


cdef inline double bilinear_interpolation(double* image, int rows, int cols,
                                          double r, double c, char mode,
                                          double cval=0):
    cdef double dr, dc
    cdef int minr, minc, maxr, maxc

    minr = <int>floor(r)
    minc = <int>floor(c)
    maxr = <int>ceil(r)
    maxc = <int>ceil(c)
    dr = r - minr
    dc = c - minc
    top = (1 - dc) * get_pixel(image, rows, cols, minr, minc, mode, cval) \
          + dc * get_pixel(image, rows, cols, minr, maxc, mode, cval)
    bottom = (1 - dc) * get_pixel(image, rows, cols, maxr, minc, mode, cval) \
             + dc * get_pixel(image, rows, cols, maxr, maxc, mode, cval)
    return (1 - dr) * top + dr * bottom


cdef inline double get_pixel(double* image, int rows, int cols, int r, int c,
                             char mode, double cval=0):
    """Get a pixel from the image, taking wrapping mode into consideration.

    Parameters
    ----------
    image : array of dtype double
        Input image.
    rows, cols: int
        Shape of image.
    r, c : int
        Position at which to get the pixel.
    mode : {'C', 'W', 'M'}
        Wrapping mode. Constant, Wrap or Mirror.
    cval : double
        Constant value to use for constant mode.

    """
    if mode == 'C':
        if (r < 0) or (r > rows - 1) or (c < 0) or (c > cols - 1):
            return cval
        else:
            return image[r * cols + c]
    else:
        return image[coord_map(rows, r, mode) * cols + coord_map(cols, c, mode)]


cdef inline int coord_map(int dim, int coord, char mode):
    """
    Wrap a coordinate, according to a given mode.

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


cdef inline tf(double x, double y, double* H, double *x_, double *y_):
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

    x_[0] = xx / zz
    y_[0] = yy / zz


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
    mode : {'constant', 'mirror', 'wrap'}
        How to handle values outside the image borders.
    cval : string
        Used in conjunction with mode 'C' (constant), the value
        outside the image boundaries.

    """

    cdef np.ndarray[dtype=np.double_t, ndim=2] img = image.astype(np.double)
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

    if output_shape is None:
        out_r = img.shape[0]
        out_c = img.shape[1]
    else:
        out_r = output_shape[0]
        out_c = output_shape[1]

    cdef np.ndarray[dtype=np.double_t, ndim=2] out = \
         np.zeros((out_r, out_c), dtype=np.double)

    cdef int tfr, tfc
    cdef double r, c
    cdef int rows = img.shape[0]
    cdef int cols = img.shape[1]

    for tfr in range(out_r):
        for tfc in range(out_c):
            tf(tfc, tfr, <double*>M.data, &c, &r)
            out[tfr, tfc] = bilinear_interpolation(<double*>img.data, rows,
                                                   cols, r, c, mode_c)

    return out
