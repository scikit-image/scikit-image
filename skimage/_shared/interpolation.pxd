#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
from libc.math cimport ceil, floor


cdef inline Py_ssize_t round(double r):
    return <Py_ssize_t>((r + 0.5) if (r > 0.0) else (r - 0.5))


cdef inline double nearest_neighbour_interpolation(double* image, Py_ssize_t rows,
                                                   Py_ssize_t cols, double r,
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

    return get_pixel2d(image, rows, cols, round(r), round(c), mode, cval)


cdef inline double bilinear_interpolation(double* image, Py_ssize_t rows,
                                          Py_ssize_t cols, double r, double c,
                                          char mode, double cval):
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
    cdef Py_ssize_t minr, minc, maxr, maxc

    minr = <Py_ssize_t>floor(r)
    minc = <Py_ssize_t>floor(c)
    maxr = <Py_ssize_t>ceil(r)
    maxc = <Py_ssize_t>ceil(c)
    dr = r - minr
    dc = c - minc
    top = (1 - dc) * get_pixel2d(image, rows, cols, minr, minc, mode, cval) \
          + dc * get_pixel2d(image, rows, cols, minr, maxc, mode, cval)
    bottom = (1 - dc) * get_pixel2d(image, rows, cols, maxr, minc, mode, cval) \
             + dc * get_pixel2d(image, rows, cols, maxr, maxc, mode, cval)
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


cdef inline double biquadratic_interpolation(double* image, Py_ssize_t rows,
                                             Py_ssize_t cols, double r, double c,
                                             char mode, double cval):
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

    cdef Py_ssize_t r0 = round(r)
    cdef Py_ssize_t c0 = round(c)
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

    cdef double fc[3]
    cdef double fr[3]

    cdef Py_ssize_t pr, pc

    # row-wise cubic interpolation
    for pr in range(r0, r0 + 3):
        for pc in range(c0, c0 + 3):
            fc[pc - c0] = get_pixel2d(image, rows, cols, pr, pc, mode, cval)
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


cdef inline double bicubic_interpolation(double* image, Py_ssize_t rows,
                                         Py_ssize_t cols, double r, double c,
                                         char mode, double cval):
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

    cdef Py_ssize_t r0 = <Py_ssize_t>r - 1
    cdef Py_ssize_t c0 = <Py_ssize_t>c - 1
    if r < 0:
        r0 -= 1
    if c < 0:
        c0 -= 1
    # scale position to range [0, 1]
    cdef double xr = (r - r0) / 3
    cdef double xc = (c - c0) / 3

    cdef double fc[4]
    cdef double fr[4]

    cdef Py_ssize_t pr, pc

    # row-wise cubic interpolation
    for pr in range(r0, r0 + 4):
        for pc in range(c0, c0 + 4):
            fc[pc - c0] = get_pixel2d(image, rows, cols, pr, pc, mode, cval)
        fr[pr - r0] = cubic_interpolation(xc, fc)

    # cubic interpolation for interpolated values of each row
    return cubic_interpolation(xr, fr)


cdef inline double get_pixel2d(double* image, Py_ssize_t rows, Py_ssize_t cols,
                               Py_ssize_t r, Py_ssize_t c, char mode, double cval):
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


cdef inline double get_pixel3d(double* image, Py_ssize_t rows, Py_ssize_t cols,
                               Py_ssize_t dims, Py_ssize_t r, Py_ssize_t c, Py_ssize_t d,
                               char mode, double cval):
    """Get a pixel from the image, taking wrapping mode into consideration.

    Parameters
    ----------
    image : double array
        Input image.
    rows, cols, dims : int
        Shape of image.
    r, c, d : int
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
            return image[r * cols * dims + c * dims + d]
    else:
        return image[coord_map(rows, r, mode) * cols * dims
                     + coord_map(cols, c, mode) * dims
                     + d]


cdef inline Py_ssize_t coord_map(Py_ssize_t dim, Py_ssize_t coord, char mode):
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
            if <Py_ssize_t>(-coord / dim) % 2 != 0:
                return dim - <Py_ssize_t>(-coord % dim)
            else:
                return <Py_ssize_t>(-coord % dim)
        elif coord > dim:
            if <Py_ssize_t>(coord / dim) % 2 != 0:
                return <Py_ssize_t>(dim - (coord % dim))
            else:
                return <Py_ssize_t>(coord % dim)
    elif mode == 'W': # wrap
        if coord < 0:
            return <Py_ssize_t>(dim - (-coord % dim))
        elif coord > dim:
            return <Py_ssize_t>(coord % dim)
    elif mode == 'N': # nearest
        if coord < 0:
            return 0
        elif coord > dim:
            return dim

    return coord
