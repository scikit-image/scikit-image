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

import numpy as np
cimport numpy as np
from .fused_numerics cimport np_real_numeric, np_floats

cdef inline Py_ssize_t round(np_floats r) nogil:
    return <Py_ssize_t>(
        (r + <np_floats>0.5) if (r > <np_floats>0.0) else (r - <np_floats>0.5)
    )

cdef inline Py_ssize_t fmax(Py_ssize_t one, Py_ssize_t two) nogil:
    return one if one > two else two

cdef inline Py_ssize_t fmin(Py_ssize_t one, Py_ssize_t two) nogil:
    return one if one < two else two

# Redefine np_real_numeric to force cross type compilation
# this allows the output type to be different than the input dtype
# https://cython.readthedocs.io/en/latest/src/userguide/fusedtypes.html#fused-types-and-arrays
ctypedef fused np_real_numeric_out:
    np_real_numeric

cdef inline void nearest_neighbour_interpolation(
        np_real_numeric* image, Py_ssize_t rows, Py_ssize_t cols,
        np_floats r, np_floats c, char mode, np_real_numeric cval,
        np_real_numeric_out* out) nogil:
    """Nearest neighbour interpolation at a given position in the image.

    Parameters
    ----------
    image : numeric array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : np_float
        Position at which to interpolate.
    mode : {'C', 'W', 'S', 'E', 'R'}
        Wrapping mode. Constant, Wrap, Symmetric, Edge or Reflect.
    cval : numeric
        Constant value to use for constant mode.

    Returns
    -------
    value : np_float
        Interpolated value.

    """

    out[0] = <np_real_numeric_out>get_pixel2d(
        image, rows, cols, round(r), round(c), mode, cval)


cdef inline void bilinear_interpolation(
        np_real_numeric* image, Py_ssize_t rows, Py_ssize_t cols,
        np_floats r, np_floats c, char mode, np_real_numeric cval,
        np_real_numeric_out* out) nogil:
    """Bilinear interpolation at a given position in the image.

    Parameters
    ----------
    image : numeric array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : np_float
        Position at which to interpolate.
    mode : {'C', 'W', 'S', 'E', 'R'}
        Wrapping mode. Constant, Wrap, Symmetric, Edge or Reflect.
    cval : numeric
        Constant value to use for constant mode.

    Returns
    -------
    value : numeric
        Interpolated value.

    """
    cdef np_floats dr, dc
    cdef long minr, minc, maxr, maxc

    minr = <long>floor(r)
    minc = <long>floor(c)
    maxr = <long>ceil(r)
    maxc = <long>ceil(c)
    dr = r - minr
    dc = c - minc

    cdef np.float64_t top
    cdef np.float64_t bottom

    cdef np_real_numeric top_left = get_pixel2d(image, rows, cols, minr, minc, mode, cval)
    cdef np_real_numeric top_right = get_pixel2d(image, rows, cols, minr, maxc, mode, cval)
    cdef np_real_numeric bottom_left = get_pixel2d(image, rows, cols, maxr, minc, mode, cval)
    cdef np_real_numeric bottom_right = get_pixel2d(image, rows, cols, maxr, maxc, mode, cval)

    top = (1 - dc) * top_left + dc * top_right
    bottom = (1 - dc) * bottom_left + dc * bottom_right
    out[0] = <np_real_numeric_out> ((1 - dr) * top + dr * bottom)

cdef inline np_floats quadratic_interpolation(np_floats x,
                                              np_real_numeric[3] f) nogil:
    """WARNING: Do not use, not implemented correctly.

    Quadratic interpolation.

    Parameters
    ----------
    x : np_float
        Position in the interval [0, 2].
    f : real numeric[3]
        Function values at positions [0, 2].

    Returns
    -------
    value : np_float
        Interpolated value to be used in biquadratic_interpolation.

    """
    return (x * f[2] * (x - 1)) / 2 - \
                x * f[1] * (x - 2) + \
                    (f[0] * (x - 1) * (x - 2)) / 2


cdef inline void biquadratic_interpolation(
        np_real_numeric* image, Py_ssize_t rows, Py_ssize_t cols,
        np_floats r, np_floats c, char mode, np_real_numeric cval,
        np_real_numeric_out* out) nogil:
    """WARNING: Do not use, not implemented correctly.

    Biquadratic interpolation at a given position in the image.

    Parameters
    ----------
    image : numeric array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : np_float
        Position at which to interpolate.
    mode : {'C', 'W', 'S', 'E', 'R'}
        Wrapping mode. Constant, Wrap, Symmetric, Edge or Reflect.
    cval : numeric
        Constant value to use for constant mode.

    Returns
    -------
    out : np_real_numeric
        Interpolated value.

    """

    cdef long r0 = <long>round(r) - 1
    cdef long c0 = <long>round(c) - 1

    cdef np_floats xr = r - r0
    cdef np_floats xc = c - c0

    cdef np_real_numeric fc[3]
    cdef np_floats fr[3]

    cdef long pr, pc

    # row-wise cubic interpolation
    for pr in range(3):
        for pc in range(3):
            fc[pc] = get_pixel2d(image, rows, cols,
                                 r0 + pr, c0 + pc, mode, cval)
        fr[pr] = quadratic_interpolation(xc, fc)

    out[0] = <np_real_numeric_out>quadratic_interpolation(xr, fr)


cdef inline np_floats cubic_interpolation(np_floats x, np_real_numeric[4] f) nogil:
    """Cubic interpolation.

    Parameters
    ----------
    x : np_float
        Position in the interval [0, 1].
    f : real numeric[4]
        Function values at positions [-1, 0, 1, 2].

    Returns
    -------
    value : np_float
        Interpolated value to be used in bicubic_interpolation.

    """

    # Explicitly cast a floating point literal to the other operand's type
    # to prevent promoting operands unnecessarily to double precision
    return (
        f[1] + <np_floats>0.5 * x * (
            f[2] - f[0] + x * (
                <np_floats>2.0 * f[0] -
                <np_floats>5.0 * f[1] +
                <np_floats>4.0 * f[2] - f[3] + x * (
                    <np_floats>3.0 * (f[1] - f[2]) + f[3] - f[0]
                )
            )
        )
    )


cdef inline void bicubic_interpolation(np_real_numeric* image,
                                       Py_ssize_t rows, Py_ssize_t cols,
                                       np_floats r, np_floats c, char mode,
                                       np_real_numeric cval,
                                       np_real_numeric_out* out) nogil:
    """Bicubic interpolation at a given position in the image.

    Interpolation using Catmull-Rom splines, based on the bicubic convolution
    algorithm described in [1]_.

    Parameters
    ----------
    image : numeric array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : np_float
        Position at which to interpolate.
    mode : {'C', 'W', 'S', 'E', 'R'}
        Wrapping mode. Constant, Wrap, Symmetric, Edge or Reflect.
    cval : numeric
        Constant value to use for constant mode.

    Returns
    -------
    out : np_real_numeric
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
    cdef np_floats xr = r - r0
    cdef np_floats xc = c - c0

    r0 -= 1
    c0 -= 1

    cdef np_real_numeric fc[4]
    cdef np_floats fr[4]
    cdef long pr, pc

    # row-wise cubic interpolation
    for pr in range(4):
        for pc in range(4):
            fc[pc] = get_pixel2d(image, rows, cols, pr + r0, pc + c0, mode, cval)
        fr[pr] = cubic_interpolation(xc, fc)

    out[0] = <np_real_numeric_out>cubic_interpolation(xr, fr)

cdef inline np_real_numeric get_pixel2d(np_real_numeric* image,
                                        Py_ssize_t rows, Py_ssize_t cols,
                                        long r, long c, char mode,
                                        np_real_numeric cval) nogil:
    """Get a pixel from the image, taking wrapping mode into consideration.

    Parameters
    ----------
    image : numeric array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : int
        Position at which to get the pixel.
    mode : {'C', 'W', 'S', 'E', 'R'}
        Wrapping mode. Constant, Wrap, Symmetric, Edge or Reflect.
    cval : numeric
        Constant value to use for constant mode.

    Returns
    -------
    value : numeric
        Pixel value at given position.

    """
    if mode == b'C':
        if (r < 0) or (r >= rows) or (c < 0) or (c >= cols):
            return cval
        else:
            return image[r * cols + c]
    else:
        return <np_real_numeric>(image[coord_map(rows, r, mode) * cols +
                                       coord_map(cols, c, mode)])


cdef inline np_real_numeric get_pixel3d(np_real_numeric* image,
                                        Py_ssize_t rows, Py_ssize_t cols,
                                        Py_ssize_t dims, Py_ssize_t r,
                                        Py_ssize_t c, Py_ssize_t d, char mode,
                                        np_real_numeric cval) nogil:
    """Get a pixel from the image, taking wrapping mode into consideration.

    Parameters
    ----------
    image : numeric array
        Input image.
    rows, cols, dims : int
        Shape of image.
    r, c, d : int
        Position at which to get the pixel.
    mode : {'C', 'W', 'S', 'E', 'R'}
        Wrapping mode. Constant, Wrap, Symmetric, Edge or Reflect.
    cval : numeric
        Constant value to use for constant mode.

    Returns
    -------
    out : np_real_numeric
        Pixel value at given position.
    """
    if mode == b'C':
        if (r < 0) or (r >= rows) or (c < 0) or (c >= cols):
            return cval
        else:
            return image[r * cols * dims + c * dims + d]
    else:
        return image[coord_map(rows, r, mode) * cols * dims +
                     coord_map(cols, c, mode) * dims +
                     coord_map(dims, d, mode)]


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
    cdef Py_ssize_t cmax = dim - 1
    if mode == b'S': # symmetric
        if coord < 0:
            coord = -coord - 1
        if coord > cmax:
            if <Py_ssize_t>(coord / dim) % 2 != 0:
                return <Py_ssize_t>(cmax - (coord % dim))
            else:
                return <Py_ssize_t>(coord % dim)
    elif mode == b'W': # wrap
        if coord < 0:
            return <Py_ssize_t>(cmax - ((-coord - 1) % dim))
        elif coord > cmax:
            return <Py_ssize_t>(coord % dim)
    elif mode == b'E': # edge
        if coord < 0:
            return 0
        elif coord > cmax:
            return cmax
    elif mode == b'R': # reflect (mirror)
        if dim == 1:
            return 0
        elif coord < 0:
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
