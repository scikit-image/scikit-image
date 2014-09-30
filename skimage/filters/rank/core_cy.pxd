from numpy cimport uint8_t, uint16_t, double_t


ctypedef fused dtype_t:
    uint8_t
    uint16_t

ctypedef fused dtype_t_out:
    uint8_t
    uint16_t
    double_t


cdef dtype_t _max(dtype_t a, dtype_t b)
cdef dtype_t _min(dtype_t a, dtype_t b)


cdef void _core(void kernel(dtype_t_out*, Py_ssize_t, Py_ssize_t*, double,
                            dtype_t, Py_ssize_t, Py_ssize_t, double,
                            double, Py_ssize_t, Py_ssize_t),
                dtype_t[:, ::1] image,
                char[:, ::1] selem,
                char[:, ::1] mask,
                dtype_t_out[:, :, ::1] out,
                signed char shift_x, signed char shift_y,
                double p0, double p1,
                Py_ssize_t s0, Py_ssize_t s1,
                Py_ssize_t max_bin) except *
