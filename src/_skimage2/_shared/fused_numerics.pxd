cimport numpy as cnp
import numpy as np

ctypedef fused np_ints:
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t

ctypedef fused np_uints:
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t

ctypedef fused np_anyint:
    np_uints
    np_ints

ctypedef fused np_floats:
    cnp.float32_t
    cnp.float64_t

ctypedef fused np_complexes:
    cnp.complex64_t
    cnp.complex128_t

ctypedef fused np_real_numeric:
    np_anyint
    np_floats

ctypedef fused np_numeric:
    np_real_numeric
    np_complexes
