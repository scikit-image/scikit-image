#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as cnp
from .._shared.interpolation cimport (nearest_neighbour_interpolation,
                                      bilinear_interpolation,
                                      biquadratic_interpolation,
                                      bicubic_interpolation)


cdef inline void _transform_metric(double x, double y, double* H,
                                   double *x_, double *y_) nogil:
    """Apply a metric transformation to a coordinate.

    Parameters
    ----------
    x, y : double
        Input coordinate.
    H : (3,3) *double
        Transformation matrix.
    x_, y_ : *double
        Output coordinate.

    """
    x_[0] = H[0] * x + H[2]
    y_[0] = H[4] * y + H[5]


cdef inline void _transform_affine(double x, double y, double* H,
                                   double *x_, double *y_) nogil:
    """Apply an affine transformation to a coordinate.

    Parameters
    ----------
    x, y : double
        Input coordinate.
    H : (3,3) *double
        Transformation matrix.
    x_, y_ : *double
        Output coordinate.

    """
    x_[0] = H[0] * x + H[1] * y + H[2]
    y_[0] = H[3] * x + H[4] * y + H[5]


cdef inline void _transform_projective(double x, double y, double* H,
                                       double *x_, double *y_) nogil:
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
    cdef double z_
    z_ = H[6] * x + H[7] * y + H[8]
    x_[0] = (H[0] * x + H[1] * y + H[2]) / z_
    y_[0] = (H[3] * x + H[4] * y + H[5]) / z_


def _warp_fast(cnp.ndarray image, cnp.ndarray H, output_shape=None,
               int order=1, mode='constant', double cval=0):
    """Projective transformation (homography).

    Perform a projective transformation (homography) of a
    floating point image, using interpolation.

    For each pixel, given its homogeneous coordinate :math:`\mathbf{x}
    = [x, y, 1]^T`, its target position is calculated by multiplying
    with the given matrix, :math:`H`, to give :math:`H \mathbf{x}`.
    E.g., to rotate by theta degrees clockwise, the matrix should be::

      [[cos(theta) -sin(theta) 0]
       [sin(theta)  cos(theta) 0]
       [0            0         1]]

    or, to translate x by 10 and y by 20::

      [[1 0 10]
       [0 1 20]
       [0 0 1 ]].

    Parameters
    ----------
    image : 2-D array
        Input image.
    H : array of shape ``(3, 3)``
        Transformation matrix H that defines the homography.
    output_shape : tuple (rows, cols), optional
        Shape of the output image generated (default None).
    order : {0, 1, 2, 3}, optional
        Order of interpolation::
        * 0: Nearest-neighbor
        * 1: Bi-linear (default)
        * 2: Bi-quadratic
        * 3: Bi-cubic
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : string, optional (default 0)
        Used in conjunction with mode 'C' (constant), the value
        outside the image boundaries.

    Notes
    -----
    Modes 'reflect' and 'symmetric' are similar, but differ in whether the edge
    pixels are duplicated during the reflection.  As an example, if an array
    has values [0, 1, 2] and was padded to the right by four values using
    symmetric, the result would be [0, 1, 2, 2, 1, 0, 0], while for reflect it
    would be [0, 1, 2, 1, 0, 1, 2].

    """

    cdef double[:, ::1] img = np.ascontiguousarray(image, dtype=np.double)
    cdef double[:, ::1] M = np.ascontiguousarray(H)

    if mode not in ('constant', 'wrap', 'symmetric', 'reflect', 'edge'):
        raise ValueError("Invalid mode specified.  Please use `constant`, "
                         "`edge`, `wrap`, `reflect` or `symmetric`.")
    cdef char mode_c = ord(mode[0].upper())

    cdef Py_ssize_t out_r, out_c
    if output_shape is None:
        out_r = int(img.shape[0])
        out_c = int(img.shape[1])
    else:
        out_r = int(output_shape[0])
        out_c = int(output_shape[1])

    cdef double[:, ::1] out = np.zeros((out_r, out_c), dtype=np.double)

    cdef Py_ssize_t tfr, tfc
    cdef double r, c
    cdef Py_ssize_t rows = img.shape[0]
    cdef Py_ssize_t cols = img.shape[1]

    cdef void (*transform_func)(double, double, double*, double*, double*) nogil
    if M[2, 0] == 0 and M[2, 1] == 0 and M[2, 2] == 1:
        if M[0, 1] == 0 and M[1, 0] == 0:
            transform_func = _transform_metric
        else:
            transform_func = _transform_affine
    else:
        transform_func = _transform_projective

    cdef double (*interp_func)(double*, Py_ssize_t, Py_ssize_t, double, double,
                               char, double) nogil
    if order == 0:
        interp_func = nearest_neighbour_interpolation
    elif order == 1:
        interp_func = bilinear_interpolation
    elif order == 2:
        interp_func = biquadratic_interpolation
    elif order == 3:
        interp_func = bicubic_interpolation
    else:
        raise ValueError("Unsupported interpolation order", order)

    with nogil:
        for tfr in range(out_r):
            for tfc in range(out_c):
                transform_func(tfc, tfr, &M[0, 0], &c, &r)
                out[tfr, tfc] = interp_func(&img[0, 0], rows, cols, r, c,
                                            mode_c, cval)

    return np.asarray(out)
