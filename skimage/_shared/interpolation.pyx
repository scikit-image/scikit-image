#cython: cdivison=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
from libc.math cimport ceil, floor


cdef inline double bilinear_interpolation(double* image, int rows, int cols,
                                          double r, double c, char mode,
                                          double cval=0):
    """Bilinear interpolation at a given position in the image.

    Parameters
    ----------
    image : double array
        Input image.
    rows, cols: int
        Shape of image.
    r, c : int
        Position at which to interpolate.
    mode : {'C', 'W', 'M'}
        Wrapping mode. Constant, Wrap or Mirror.
    cval : double
        Constant value to use for constant mode.

    """
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
    image : double array
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
