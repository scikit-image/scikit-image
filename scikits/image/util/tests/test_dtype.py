import numpy as np
from numpy.testing import assert_equal
from scikits.image import img_as_int, img_as_float, \
                          img_as_uint, img_as_ubyte

dtype_range = {np.uint8: (0, 255),
               np.uint16: (0, 65535),
               np.int8: (-128, 127),
               np.int16: (-32768, 32767),
               np.float32: (0, 1),
               np.float64: (0, 1)}

def _verify_range(msg, x, vmin, vmax):
    assert_equal(x[0], vmin)
    assert_equal(x[-1], vmax)

def test_range():
    for dtype in dtype_range:
        imin, imax = dtype_range[dtype]
        x = np.linspace(imin, imax, 10).astype(dtype)

        for (f, dt) in [(img_as_int, np.int16),
                        (img_as_float, np.float64),
                        (img_as_uint, np.uint16),
                        (img_as_ubyte, np.ubyte)]:
            y = f(x)

            omin, omax = dtype_range[dt]

            if imin == 0:
                omin = 0

            yield _verify_range, \
                  "From %s to %s" % (np.dtype(dtype), np.dtype(dt)), \
                  y, omin, omax

if __name__ == '__main__':
    np.testing.run_module_suite()

