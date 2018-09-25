/* A fast approximation of the exponential function.
 * Reference [1]: https://schraudolph.org/pubs/Schraudolph99.pdf
 * Reference [2]: http://dx.doi.org/10.1162/089976600300015033
 * Additional improvements by Leonid Bloch. */

/* use just EXP_A = 1512775 for integer version, to avoid FP calculations */
#define EXP_A (1512775.3951951856938)  /* 2^20*ln2 */
/* For min. RMS error */
#define EXP_BC 1072632447              /* 1023*2^20 - 60801 */
/* For min. max. relative error */
/* #define EXP_BC 1072647449 */        /* 1023*2^20 - 45799 */
/* For min. mean relative error */
/* #define EXP_BC 1072625005 */        /* 1023*2^20 - 68243 */

__inline double fast_exp (double y)
{
    union
    {
        double d;
        struct { int i, j; } n;
        char t[8];
    } _eco;

    _eco.n.i = 1;

    switch(_eco.t[0]) {
        case 1:
            /* Little endian */
            _eco.n.j = (int)(EXP_A*(y)) + EXP_BC;
            _eco.n.i = 0;
            break;
        case 0:
            /* Big endian */
            _eco.n.i = (int)(EXP_A*(y)) + EXP_BC;
            _eco.n.j = 0;
            break;
    }

    return _eco.d;
}
