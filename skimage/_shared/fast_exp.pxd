# /* A fast approximation of the exponential function.
#  * Reference [1]: https://schraudolph.org/pubs/Schraudolph99.pdf
#  * Reference [2]: https://doi.org/10.1162/089976600300015033
#  * Additional improvements by Leonid Bloch. */

# /* use just EXP_A = 1512775 for integer version, to avoid FP calculations */
# /* 2^20*ln2 */
cdef double EXP_A = (1512775.3951951856938)
# /* For min. RMS error */
# /* 1023*2^20 - 60801 */
cdef int EXP_BC = 1072632447
# /* For min. max. relative error */
# /* #define EXP_BC 1072647449 */        /* 1023*2^20 - 45799 */
# /* For min. mean relative error */
# /* #define EXP_BC 1072625005 */        /* 1023*2^20 - 68243 */
from libc.stdint cimport int32_t

cdef struct _n:
    int32_t i
    int32_t j
cdef union _eco:
    double d
    _n n
    char t[8]

cdef inline double fast_exp (double y) nogil:
    cdef _eco eco
    eco.n.i = 1;

    if eco.t[0] == 1:
        # Little endian
        eco.n.j = (<int32_t>(EXP_A*(y))) + EXP_BC
        eco.n.i = 0
    else:
        # Big endian
        eco.n.i = (<int32_t>(EXP_A*(y))) + EXP_BC
        eco.n.j = 0

    return eco.d
