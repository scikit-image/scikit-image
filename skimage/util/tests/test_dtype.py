from functools import partial

import numpy as np
import itertools
import pytest

import skimage.util.dtype
from skimage.util import (
    rescale_to_float,
    rescale_to_float32,
    rescale_to_float64,
    rescale_to_int16,
    rescale_to_uint16,
    rescale_to_uint8,
)
from skimage.util.dtype import _convert

from skimage._shared._warnings import expected_warnings
from skimage._shared import testing
from skimage._shared.testing import assert_equal, parametrize, assert_stacklevel


dtype_range = {
    np.uint8: (0, 255),
    np.uint16: (0, 65535),
    np.int8: (-128, 127),
    np.int16: (-32768, 32767),
    np.float32: (-1.0, 1.0),
    np.float64: (-1.0, 1.0),
}


rescale_funcs = (
    rescale_to_int16,
    partial(rescale_to_float64, legacy_float_range=True),
    partial(rescale_to_float32, legacy_float_range=True),
    rescale_to_uint16,
    rescale_to_uint8,
)
dtypes_for_rescale_funcs = (np.int16, np.float64, np.float32, np.uint16, np.ubyte)
rescale_funcs_and_types = zip(rescale_funcs, dtypes_for_rescale_funcs)


def _verify_range(msg, x, vmin, vmax, dtype):
    assert_equal(x[0], vmin)
    assert_equal(x[-1], vmax)
    assert x.dtype == dtype


@parametrize("dtype, f_and_dt", itertools.product(dtype_range, rescale_funcs_and_types))
def test_range(dtype, f_and_dt):
    imin, imax = dtype_range[dtype]
    x = np.linspace(imin, imax, 10).astype(dtype)

    f, dt = f_and_dt

    y = f(x)

    omin, omax = dtype_range[dt]

    if imin == 0 or omin == 0:
        omin = 0
        imin = 0

    _verify_range(
        f"From {np.dtype(dtype)} to {np.dtype(dt)}", y, omin, omax, np.dtype(dt)
    )


# Add non-standard data types that are allowed by the `_convert` function.
dtype_range_extra = dtype_range.copy()
dtype_range_extra.update(
    {np.int32: (-2147483648, 2147483647), np.uint32: (0, 4294967295)}
)

dtype_pairs = [
    (np.uint8, np.uint32),
    (np.int8, np.uint32),
    (np.int8, np.int32),
    (np.int32, np.int8),
    (np.float64, np.float32),
    (np.int32, np.float32),
]


@parametrize("dtype_in, dt", dtype_pairs)
def test_range_extra_dtypes(dtype_in, dt):
    """Test code paths that are not skipped by `test_range`"""

    imin, imax = dtype_range_extra[dtype_in]
    x = np.linspace(imin, imax, 10).astype(dtype_in)

    y = _convert(x, dt, legacy_float_range=True)

    omin, omax = dtype_range_extra[dt]
    _verify_range(
        f"From {np.dtype(dtype_in)} to {np.dtype(dt)}", y, omin, omax, np.dtype(dt)
    )


def test_downcast():
    x = np.arange(10).astype(np.uint64)
    with expected_warnings(['Downcasting']):
        y = rescale_to_int16(x)
    assert np.allclose(y, x.astype(np.int16))
    assert y.dtype == np.int16, y.dtype


def test_float_out_of_range():
    too_high = np.array([2], dtype=np.float32)
    with testing.raises(ValueError):
        rescale_to_int16(too_high)
    too_low = np.array([-2], dtype=np.float32)
    with testing.raises(ValueError):
        rescale_to_int16(too_low)


@pytest.mark.parametrize("legacy_float_range", [True, False])
def test_float_float_all_ranges(legacy_float_range):
    arr_in = np.array([[-10.0, 10.0, 1e20]], dtype=np.float32)
    np.testing.assert_array_equal(
        rescale_to_float(arr_in, legacy_float_range=legacy_float_range), arr_in
    )


@pytest.mark.parametrize("legacy_float_range", [True, False])
def test_copy(legacy_float_range):
    x = np.array([1], dtype=np.float64)
    y = rescale_to_float(x, legacy_float_range=legacy_float_range)
    z = rescale_to_float(x, force_copy=True, legacy_float_range=legacy_float_range)

    assert y is x
    assert z is not x


def test_bool():
    img_ = np.zeros((10, 10), bool)
    img8 = np.zeros((10, 10), np.bool_)
    img_[1, 1] = True
    img8[1, 1] = True
    for func, dt in [
        (rescale_to_int16, np.int16),
        (rescale_to_float, np.float64),
        (rescale_to_uint16, np.uint16),
        (rescale_to_uint8, np.ubyte),
    ]:
        converted_ = func(img_)
        assert np.sum(converted_) == dtype_range[dt][1]
        converted8 = func(img8)
        assert np.sum(converted8) == dtype_range[dt][1]


def test_clobber():
    # The `img_as_*` functions should never modify input arrays.
    for func_input_type in rescale_funcs:
        for func_output_type in rescale_funcs:
            img = np.random.rand(5, 5)

            img_in = func_input_type(img)
            img_in_before = img_in.copy()
            func_output_type(img_in)

            assert_equal(img_in, img_in_before)


@pytest.mark.parametrize("legacy_float_range", [True, False])
def test_signed_scaling_float32(legacy_float_range):
    x = np.array([-128, 127], dtype=np.int8)
    y = rescale_to_float32(x, legacy_float_range=legacy_float_range)
    assert_equal(y.max(), 1)


@pytest.mark.parametrize("legacy_float_range", [True, False])
def test_float32_passthrough(legacy_float_range):
    x = np.array([-1, 1], dtype=np.float32)
    y = rescale_to_float(x, legacy_float_range=legacy_float_range)
    assert_equal(y.dtype, x.dtype)


float_dtype_list = [
    float,
    float,
    np.float64,
    np.single,
    np.float32,
    np.float64,
    'float32',
    'float64',
]


def test_float_conversion_dtype():
    """Test any conversion from a float dtype to an other."""
    x = np.array([-1, 1])

    # Test all combinations of dtypes conversions
    dtype_combin = np.array(np.meshgrid(float_dtype_list, float_dtype_list)).T.reshape(
        -1, 2
    )

    for dtype_in, dtype_out in dtype_combin:
        x = x.astype(dtype_in)
        y = _convert(x, dtype_out)
        assert y.dtype == np.dtype(dtype_out)


def test_float_conversion_dtype_warns():
    """Test that convert issues a warning when called"""
    from skimage.util.dtype import convert

    x = np.array([-1, 1])

    # Test all combinations of dtypes conversions
    dtype_combin = np.array(np.meshgrid(float_dtype_list, float_dtype_list)).T.reshape(
        -1, 2
    )

    for dtype_in, dtype_out in dtype_combin:
        x = x.astype(dtype_in)
        with expected_warnings(["The use of this function is discouraged"]):
            y = convert(x, dtype_out)
        assert y.dtype == np.dtype(dtype_out)


def test_subclass_conversion():
    """Check subclass conversion behavior"""
    x = np.array([-1, 1])

    for dtype in float_dtype_list:
        x = x.astype(dtype)
        y = _convert(x, np.floating)
        assert y.dtype == x.dtype


def test_int_to_float():
    """Check Normalization when casting rescale_to_float from int types to float"""
    int_list = np.arange(9, dtype=np.int64)
    converted = rescale_to_float(int_list, legacy_float_range=True)
    assert np.allclose(converted, int_list * 1e-19, atol=0.0, rtol=0.1)

    ii32 = np.iinfo(np.int32)
    ii_list = np.array([ii32.min, ii32.max], dtype=np.int32)
    floats = rescale_to_float(ii_list, legacy_float_range=True)

    assert_equal(floats.max(), 1)
    assert_equal(floats.min(), -1)


def test_rescale_to_uint8_supports_npulonglong():
    # Pre NumPy <2.0.0, `data_scaled.dtype.type` is `np.ulonglong` instead of
    # np.uint64 as one might expect. This caused issues with `rescale_to_uint8` due
    # to `np.ulonglong` missing from `skimage.util.dtype._integer_types`.
    # This doesn't seem to be an issue for NumPy >=2.0.0.
    # https://github.com/scikit-image/scikit-image/issues/7385
    data = np.arange(50, dtype=np.uint64)
    data_scaled = data * 256 ** (data.dtype.itemsize - 1)
    result = rescale_to_uint8(data_scaled)
    assert result.dtype == np.uint8


@pytest.mark.parametrize("module", [skimage, skimage.util, skimage.util.dtype])
@pytest.mark.parametrize(
    "name",
    [
        "img_as_float",
        "img_as_float32",
        "img_as_float64",
        "img_as_int",
        "img_as_uint",
        "img_as_ubyte",
    ],
)
def test_deprecation_of_img_as_funcs(module, name):
    func = getattr(module, name)
    img = np.linspace(-1, 1)
    regex = "`img_as_.*` is deprecated.*Use `skimage.util.rescale_to_.*`"
    with pytest.warns(FutureWarning, match=regex) as record:
        func(img)
    assert len(record) == 1
    assert_stacklevel(record, offset=-2)


@pytest.mark.parametrize("dtype_in", [np.int8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize("dtype_out", [np.float16, np.float32, np.float64])
def test_convert_signed_to_float(dtype_in, dtype_out):
    image = np.array([np.iinfo(dtype_in).min, np.iinfo(dtype_in).max], dtype=dtype_in)
    expected = np.array([0, 1], dtype=dtype_out)
    result = _convert(image, dtype=dtype_out, legacy_float_range=False)
    assert result.dtype == dtype_out
    np.testing.assert_equal(expected, result)


@pytest.mark.parametrize("dtype_in", [np.int8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize("dtype_out", [np.float16, np.float32, np.float64])
def test_convert_signed_to_float_legacy(dtype_in, dtype_out):
    image = np.array(
        [np.iinfo(dtype_in).min, 0, np.iinfo(dtype_in).max], dtype=dtype_in
    )
    expected = np.array([-1, 0, 1], dtype=dtype_out)
    result = _convert(image, dtype=dtype_out, legacy_float_range=True)
    assert result.dtype == dtype_out
    np.testing.assert_equal(expected, result)
