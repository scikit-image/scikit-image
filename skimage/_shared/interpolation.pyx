#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
from libc.math cimport ceil, floor


cdef inline int round(double r):
    return <int>((r + 0.5) if (r > 0.0) else (r - 0.5))


cdef inline double nearest_neighbour_interpolation(double* image, int rows,
                                                   int cols, double r,
                                                   double c, char mode,
                                                   double cval):
    """Nearest neighbour interpolation at a given position in the image.

    Parameters
    ----------
    image : double array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : double
        Position at which to interpolate.
    mode : {'C', 'W', 'R', 'N'}
        Wrapping mode. Constant, Wrap, Reflect or Nearest.
    cval : double
        Constant value to use for constant mode.

    Returns
    -------
    value : double
        Interpolated value.

    """

    return get_pixel(image, rows, cols, <int>round(r), <int>round(c),
                     mode, cval)


cdef inline double bilinear_interpolation(double* image, int rows, int cols,
                                          double r, double c, char mode,
                                          double cval):
    """Bilinear interpolation at a given position in the image.

    Parameters
    ----------
    image : double array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : double
        Position at which to interpolate.
    mode : {'C', 'W', 'R', 'N'}
        Wrapping mode. Constant, Wrap, Reflect or Nearest.
    cval : double
        Constant value to use for constant mode.

    Returns
    -------
    value : double
        Interpolated value.

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


cdef inline double quadratic_interpolation(double x, double[3] f):
    """Quadratic interpolation.

    Parameters
    ----------
    x : double
        Position in the interval [-1, 1].
    f : double[4]
        Function values at positions [-1, 0, 1].

    Returns
    -------
    value : double
        Interpolated value.

    """
    return f[1] - 0.25 * (f[0] - f[2]) * x


cdef inline double biquadratic_interpolation(double* image, int rows, int cols,
                                             double r, double c, char mode,
                                             double cval):
    """Biquadratic interpolation at a given position in the image.

    Parameters
    ----------
    image : double array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : double
        Position at which to interpolate.
    mode : {'C', 'W', 'R', 'N'}
        Wrapping mode. Constant, Wrap, Reflect or Nearest.
    cval : double
        Constant value to use for constant mode.

    Returns
    -------
    value : double
        Interpolated value.

    """

    cdef int r0 = <int>round(r)
    cdef int c0 = <int>round(c)
    if r < 0:
        r0 -= 1
    if c < 0:
        c0 -= 1
    # scale position to range [-1, 1]
    cdef double xr = (r - r0) - 1
    cdef double xc = (c - c0) - 1
    if r == r0:
        xr += 1
    if c == c0:
        xc += 1

    cdef double fc[3], fr[3]

    cdef int pr, pc

    # row-wise cubic interpolation
    for pr in range(r0, r0 + 3):
        for pc in range(c0, c0 + 3):
            fc[pc - c0] = get_pixel(image, rows, cols, pr, pc, mode, cval)
        fr[pr - r0] = quadratic_interpolation(xc, fc)

    # cubic interpolation for interpolated values of each row
    return quadratic_interpolation(xr, fr)


cdef inline double cubic_interpolation(double x, double[4] f):
    """Cubic interpolation.

    Parameters
    ----------
    x : double
        Position in the interval [0, 1].
    f : double[4]
        Function values at positions [0, 1/3, 2/3, 1].

    Returns
    -------
    value : double
        Interpolated value.

    """
    return \
        f[1] + 0.5 * x * \
            (f[2] - f[0] + x * \
                (2.0 * f[0] - 5.0 * f[1] + 4.0 * f[2] - f[3] + x * \
                    (3.0 * (f[1] - f[2]) + f[3] - f[0])))


cdef inline double bicubic_interpolation(double* image, int rows, int cols,
                                         double r, double c, char mode,
                                         double cval):
    """Bicubic interpolation at a given position in the image.

    Parameters
    ----------
    image : double array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : double
        Position at which to interpolate.
    mode : {'C', 'W', 'R', 'N'}
        Wrapping mode. Constant, Wrap, Reflect or Nearest.
    cval : double
        Constant value to use for constant mode.

    Returns
    -------
    value : double
        Interpolated value.

    """

    cdef int r0 = <int>r - 1
    cdef int c0 = <int>c - 1
    if r < 0:
        r0 -= 1
    if c < 0:
        c0 -= 1
    # scale position to range [0, 1]
    cdef double xr = (r - r0) / 3
    cdef double xc = (c - c0) / 3

    cdef double fc[4], fr[4]

    cdef int pr, pc

    # row-wise cubic interpolation
    for pr in range(r0, r0 + 4):
        for pc in range(c0, c0 + 4):
            fc[pc - c0] = get_pixel(image, rows, cols, pr, pc, mode, cval)
        fr[pr - r0] = cubic_interpolation(xc, fc)

    # cubic interpolation for interpolated values of each row
    return cubic_interpolation(xr, fr)


cdef inline double get_pixel(double* image, int rows, int cols, int r, int c,
                             char mode, double cval):
    """Get a pixel from the image, taking wrapping mode into consideration.

    Parameters
    ----------
    image : double array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : int
        Position at which to get the pixel.
    mode : {'C', 'W', 'R', 'N'}
        Wrapping mode. Constant, Wrap, Reflect or Nearest.
    cval : double
        Constant value to use for constant mode.

    Returns
    -------
    value : double
        Pixel value at given position.

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
    mode : {'W', 'R', 'N'}
        Whether to wrap or reflect the coordinate if it
        falls outside [0, dim).

    """
    dim = dim - 1
    if mode == 'R': # reflect
        if coord < 0:
            # How many times times does the coordinate wrap?
            if <int>(-coord / dim) % 2 != 0:
                return dim - <int>(-coord % dim)
            else:
                return <int>(-coord % dim)
        elif coord > dim:
            if <int>(coord / dim) % 2 != 0:
                return <int>(dim - (coord % dim))
            else:
                return <int>(coord % dim)
    elif mode == 'W': # wrap
        if coord < 0:
            return <int>(dim - (-coord % dim))
        elif coord > dim:
            return <int>(coord % dim)
    elif mode == 'N': # nearest
        if coord < 0:
            return 0
        elif coord > dim:
            return dim

    return coord
