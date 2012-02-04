import numpy as np
from numpy.testing import assert_equal, assert_raises
from skimage import img_as_int, img_as_float, \
                    img_as_uint, img_as_ubyte
from skimage.util.dtype import convert


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


def test_range_extra_dtypes():
    """Test code paths that are not skipped by `test_range`"""

    # Add non-standard data types that are allowed by the `convert` function.
    dtype_range_extra = dtype_range.copy()
    dtype_range_extra.update({np.int32: (-2147483648, 2147483647),
                              np.uint32: (0, 4294967295)})

    dtype_pairs = [(np.uint8, np.uint32),
                   (np.int8, np.uint32),
                   (np.int8, np.int32),
                   (np.int32, np.int8),
                   (np.float64, np.float32),
                   (np.int32, np.float32)]

    for dtype_in, dt in dtype_pairs:
        imin, imax = dtype_range_extra[dtype_in]
        x = np.linspace(imin, imax, 10).astype(dtype_in)
        y = convert(x, dt)
        omin, omax = dtype_range_extra[dt]
        yield _verify_range, \
              "From %s to %s" % (np.dtype(dtype_in), np.dtype(dt)), \
              y, omin, omax


def test_unsupported_dtype():
    x = np.arange(10).astype(np.uint64)
    assert_raises(ValueError, img_as_int, x)


def test_float_out_of_range():
    too_high = np.array([2], dtype=np.float32)
    assert_raises(ValueError, img_as_int, too_high)
    too_low = np.array([-1], dtype=np.float32)
    assert_raises(ValueError, img_as_int, too_low)


if __name__ == '__main__':
    np.testing.run_module_suite()

