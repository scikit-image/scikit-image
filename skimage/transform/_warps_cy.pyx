#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as np
import numpy as np
from skimage._shared.interpolation cimport (nearest_neighbour_interpolation,
                                            bilinear_interpolation,
                                            biquadratic_interpolation,
                                            bicubic_interpolation)


cdef inline void _matrix_transform(double x, double y, double* H, double *x_,
                                   double *y_):
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


def _warp_fast(np.ndarray image, np.ndarray H, output_shape=None, int order=1,
               mode='constant', double cval=0):
    """Projective transformation (homography).

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
    order : {0, 1}
        Order of interpolation::
        * 0: Nearest-neighbour interpolation.
        * 1: Bilinear interpolation (default).
        * 2: Biquadratic interpolation (default).
        * 3: Bicubic interpolation.
    mode : {'constant', 'reflect', 'wrap'}
        How to handle values outside the image borders.
    cval : string
        Used in conjunction with mode 'C' (constant), the value
        outside the image boundaries.

    """

    cdef np.ndarray[dtype=np.double_t, ndim=2, mode="c"] img = \
         np.ascontiguousarray(image, dtype=np.double)
    cdef np.ndarray[dtype=np.double_t, ndim=2, mode="c"] M = \
         np.ascontiguousarray(H)

    if mode not in ('constant', 'wrap', 'reflect', 'nearest'):
        raise ValueError("Invalid mode specified.  Please use "
                         "`constant`, `nearest`, `wrap` or `reflect`.")
    cdef char mode_c = ord(mode[0].upper())

    cdef int out_r, out_c
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

    cdef double (*interp_func)(double*, int, int, double, double,
                               char, double)
    if order == 0:
        interp_func = nearest_neighbour_interpolation
    elif order == 1:
        interp_func = bilinear_interpolation
    elif order == 2:
        interp_func = biquadratic_interpolation
    elif order == 3:
        interp_func = bicubic_interpolation

    for tfr in range(out_r):
        for tfc in range(out_c):
            _matrix_transform(tfc, tfr, <double*>M.data, &c, &r)
            out[tfr, tfc] = interp_func(<double*>img.data, rows, cols, r, c,
                                        mode_c, cval)

    return out
