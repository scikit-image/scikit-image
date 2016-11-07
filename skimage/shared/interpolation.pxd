#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
"""
Note: All edge modes implemented here follow the corresponding numpy.pad
conventions.

The table below illustrates the behavior for the array [1, 2, 3, 4], if padded
by 4 values on each side:

                               pad     original    pad
    constant (with c=0) :    0 0 0 0 | 1 2 3 4 | 0 0 0 0
    wrap                :    1 2 3 4 | 1 2 3 4 | 1 2 3 4
    symmetric           :    4 3 2 1 | 1 2 3 4 | 4 3 2 1
    edge                :    1 1 1 1 | 1 2 3 4 | 4 4 4 4
    reflect             :    3 4 3 2 | 1 2 3 4 | 3 2 1 2
"""
from libc.math cimport ceil, floor


cdef inline Py_ssize_t round(double r) nogil:
    return <Py_ssize_t>((r + 0.5) if (r > 0.0) else (r - 0.5))


cdef inline double nearest_neighbour_interpolation(double* image,
                                                   Py_ssize_t rows,
                                                   Py_ssize_t cols, double r,
                                                   double c, char mode,
                                                   double cval) nogil:
    """Nearest neighbour interpolation at a given position in the image.

    Parameters
    ----------
    image : double array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : double
        Position at which to interpolate.
    mode : {'C', 'W', 'S', 'E', 'R'}
        Wrapping mode. Constant, Wrap, Symmetric, Edge or Reflect.
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
                                          char mode, double cval) nogil:
    """Bilinear interpolation at a given position in the image.

    Parameters
    ----------
    image : double array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : double
        Position at which to interpolate.
    mode : {'C', 'W', 'S', 'E', 'R'}
        Wrapping mode. Constant, Wrap, Symmetric, Edge or Reflect.
    cval : double
        Constant value to use for constant mode.

    Returns
    -------
    value : double
        Interpolated value.

    """
    cdef double dr, dc
    cdef long minr, minc, maxr, maxc

    minr = <long>floor(r)
    minc = <long>floor(c)
    maxr = <long>ceil(r)
    maxc = <long>ceil(c)
    dr = r - minr
    dc = c - minc
    top = (1 - dc) * get_pixel2d(image, rows, cols, minr, minc, mode, cval) \
          + dc * get_pixel2d(image, rows, cols, minr, maxc, mode, cval)
    bottom = (1 - dc) * get_pixel2d(image, rows, cols, maxr, minc, mode,
                                    cval) \
             + dc * get_pixel2d(image, rows, cols, maxr, maxc, mode, cval)
    return (1 - dr) * top + dr * bottom


cdef inline double quadratic_interpolation(double x, double[3] f) nogil:
    """WARNING: Do not use, not implemented correctly.

    Quadratic interpolation.

    Parameters
    ----------
    x : double
        Position in the interval [0, 2].
    f : double[3]
        Function values at positions [0, 2].

    Returns
    -------
    value : double
        Interpolated value.

    """
    return (x * f[2] * (x - 1)) / 2 - \
                x * f[1] * (x - 2) + \
                    (f[0] * (x - 1) * (x - 2)) / 2


cdef inline double biquadratic_interpolation(double* image, Py_ssize_t rows,
                                             Py_ssize_t cols, double r,
                                             double c, char mode,
                                             double cval) nogil:
    """WARNING: Do not use, not implemented correctly.

    Biquadratic interpolation at a given position in the image.

    Parameters
    ----------
    image : double array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : double
        Position at which to interpolate.
    mode : {'C', 'W', 'S', 'E', 'R'}
        Wrapping mode. Constant, Wrap, Symmetric, Edge or Reflect.
    cval : double
        Constant value to use for constant mode.

    Returns
    -------
    value : double
        Interpolated value.

    """

    cdef long r0 = <long>round(r) - 1
    cdef long c0 = <long>round(c) - 1

    cdef double xr = r - r0
    cdef double xc = c - c0

    cdef double fc[3]
    cdef double fr[3]

    cdef long pr, pc

    # row-wise cubic interpolation
    for pr in range(3):
        for pc in range(3):
            fc[pc] = get_pixel2d(image, rows, cols,
                                 r0 + pr, c0 + pc, mode, cval)
        fr[pr] = quadratic_interpolation(xc, fc)

    # cubic interpolation for interpolated values of each row
    return quadratic_interpolation(xr, fr)


cdef inline double cubic_interpolation(double x, double[4] f) nogil:
    """Cubic interpolation.

    Parameters
    ----------
    x : double
        Position in the interval [0, 1].
    f : double[4]
        Function values at positions [-1, 0, 1, 2].

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
                                         char mode, double cval) nogil:
    """Bicubic interpolation at a given position in the image.

    Interpolation using Catmull-Rom splines, based on the bicubic convolution
    algorithm described in [1]_.

    Parameters
    ----------
    image : double array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : double
        Position at which to interpolate.
    mode : {'C', 'W', 'S', 'E', 'R'}
        Wrapping mode. Constant, Wrap, Symmetric, Edge or Reflect.
    cval : double
        Constant value to use for constant mode.

    Returns
    -------
    value : double
        Interpolated value.

    References
    ----------
    .. [1] R. Keys, (1981). "Cubic convolution interpolation for digital image
           processing". IEEE Transactions on Signal Processing, Acoustics,
           Speech, and Signal Processing 29 (6): 1153–1160.

    """

    cdef long r0 = <long>floor(r)
    cdef long c0 = <long>floor(c)

    # scale position to range [0, 1]
    cdef double xr = r - r0
    cdef double xc = c - c0

    r0 -= 1
    c0 -= 1

    cdef double fc[4]
    cdef double fr[4]

    cdef long pr, pc

    # row-wise cubic interpolation
    for pr in range(4):
        for pc in range(4):
            fc[pc] = get_pixel2d(image, rows, cols, pr + r0, pc + c0, mode, cval)
        fr[pr] = cubic_interpolation(xc, fc)

    # cubic interpolation for interpolated values of each row
    return cubic_interpolation(xr, fr)


cdef inline double get_pixel2d(double* image, Py_ssize_t rows, Py_ssize_t cols,
                               long r, long c, char mode,
                               double cval) nogil:
    """Get a pixel from the image, taking wrapping mode into consideration.

    Parameters
    ----------
    image : double array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : int
        Position at which to get the pixel.
    mode : {'C', 'W', 'S', 'E', 'R'}
        Wrapping mode. Constant, Wrap, Symmetric, Edge or Reflect.
    cval : double
        Constant value to use for constant mode.

    Returns
    -------
    value : double
        Pixel value at given position.

    """
    if mode == 'C':
        if (r < 0) or (r >= rows) or (c < 0) or (c >= cols):
            return cval
        else:
            return image[r * cols + c]
    else:
        return image[coord_map(rows, r, mode) * cols + coord_map(cols, c, mode)]


cdef inline double get_pixel3d(double* image, Py_ssize_t rows, Py_ssize_t cols,
                               Py_ssize_t dims, long r, long c,
                               long d, char mode, double cval) nogil:
    """Get a pixel from the image, taking wrapping mode into consideration.

    Parameters
    ----------
    image : double array
        Input image.
    rows, cols, dims : int
        Shape of image.
    r, c, d : int
        Position at which to get the pixel.
    mode : {'C', 'W', 'S', 'E', 'R'}
        Wrapping mode. Constant, Wrap, Symmetric, Edge or Reflect.
    cval : double
        Constant value to use for constant mode.

    Returns
    -------
    value : double
        Pixel value at given position.

    """
    if mode == 'C':
        if (r < 0) or (r >= rows) or (c < 0) or (c >= cols):
            return cval
        else:
            return image[r * cols * dims + c * dims + d]
    else:
        return image[coord_map(rows, r, mode) * cols * dims
                     + coord_map(cols, c, mode) * dims
                     + coord_map(dims, d, mode)]


cdef inline Py_ssize_t coord_map(Py_ssize_t dim, long coord, char mode) nogil:
    """Wrap a coordinate, according to a given mode.

    Parameters
    ----------
    dim : int
        Maximum coordinate.
    coord : int
        Coord provided by user.  May be < 0 or > dim.
    mode : {'W', 'S', 'R', 'E'}
        Whether to wrap, symmetric reflect, reflect or use the nearest
        coordinate if `coord` falls outside [0, dim).
    """
    cdef Py_ssize_t cmax
    cmax = dim - 1
    if mode == 'S': # symmetric
        if coord < 0:
            coord = -coord - 1
        if coord > cmax:
            if <Py_ssize_t>(coord / dim) % 2 != 0:
                return <Py_ssize_t>(cmax - (coord % dim))
            else:
                return <Py_ssize_t>(coord % dim)
    elif mode == 'W': # wrap
        if coord < 0:
            return <Py_ssize_t>(cmax - ((-coord - 1) % dim))
        elif coord > cmax:
            return <Py_ssize_t>(coord % dim)
    elif mode == 'E': # edge
        if coord < 0:
            return 0
        elif coord > cmax:
            return cmax
    elif mode == 'R': # reflect (mirror)
        if coord < 0:
            # How many times times does the coordinate wrap?
            if <Py_ssize_t>(-coord / cmax) % 2 != 0:
                return cmax - <Py_ssize_t>(-coord % cmax)
            else:
                return <Py_ssize_t>(-coord % cmax)
        elif coord > cmax:
            if <Py_ssize_t>(coord / cmax) % 2 != 0:
                return <Py_ssize_t>(cmax - (coord % cmax))
            else:
                return <Py_ssize_t>(coord % cmax)
    return coord
