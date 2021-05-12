import numpy as np

from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import parametrize
from skimage._shared.utils import _supported_float_type
from skimage.filters import unsharp_mask


@parametrize("shape,multichannel",
             [((29,), False),
              ((40, 4), True),
              ((32, 32), False),
              ((29, 31, 3), True),
              ((13, 17, 4, 8), False)])
@parametrize("dtype", [np.uint8, np.int8,
                       np.uint16, np.int16,
                       np.uint32, np.int32,
                       np.uint64, np.int64,
                       np.float32, np.float64])
@parametrize("radius", [0, 0.1, 2.0])
@parametrize("amount", [0.0, 0.5, 2.0, -1.0])
@parametrize("offset", [-1.0, 0.0, 1.0])
@parametrize("preserve", [False, True])
def test_unsharp_masking_output_type_and_shape(
        radius, amount, shape, multichannel, dtype, offset, preserve):
    array = np.random.random(shape)
    array = ((array + offset) * 128).astype(dtype)
    if (preserve is False) and (dtype in [np.float32, np.float64]):
        array /= max(np.abs(array).max(), 1.0)
    channel_axis = -1 if multichannel else None
    output = unsharp_mask(array, radius, amount, preserve_range=preserve,
                          channel_axis=channel_axis)
    assert output.dtype in [np.float32, np.float64]
    assert output.shape == shape


@parametrize("shape,multichannel",
             [((32, 32), False),
              ((15, 15, 2), True),
              ((17, 19, 3), True)])
@parametrize("radius", [(0.0, 0.0), (1.0, 1.0), (2.0, 1.5)])
@parametrize("preserve", [False, True])
def test_unsharp_masking_with_different_radii(radius, shape,
                                              multichannel, preserve):
    amount = 1.0
    dtype = np.float64
    array = (np.random.random(shape) * 96).astype(dtype)
    if preserve is False:
        array /= max(np.abs(array).max(), 1.0)
    channel_axis = -1 if multichannel else None
    output = unsharp_mask(array, radius, amount, preserve_range=preserve,
                          channel_axis=channel_axis)
    assert output.dtype in [np.float32, np.float64]
    assert output.shape == shape


@parametrize("shape,channel_axis",
             [((16, 16), None),
              ((15, 15, 2), -1),
              ((13, 17, 3), -1),
              ((2, 15, 15), 0),
              ((3, 13, 17), 0)])
@parametrize("offset", [-5, 0, 5])
@parametrize("preserve", [False, True])
def test_unsharp_masking_with_different_ranges_deprecated(shape, offset,
                                                          channel_axis,
                                                          preserve):
    radius = 2.0
    amount = 1.0
    dtype = np.int16
    array = (np.random.random(shape) * 5 + offset).astype(dtype)
    negative = np.any(array < 0)
    output = unsharp_mask(array, radius, amount, preserve_range=preserve,
                          channel_axis=channel_axis)
    if preserve is False:
        assert np.any(output <= 1)
        assert np.any(output >= -1)
        if negative is False:
            assert np.any(output >= 0)
    assert output.dtype in [np.float32, np.float64]
    assert output.shape == shape


@parametrize("shape,multichannel",
             [((16, 16), False),
              ((15, 15, 2), True),
              ((13, 17, 3), True)])
@parametrize("offset", [-5, 0, 5])
@parametrize("preserve", [False, True])
def test_unsharp_masking_with_different_ranges_deprecated(shape, offset,
                                                          multichannel,
                                                          preserve):
    radius = 2.0
    amount = 1.0
    dtype = np.int16
    array = (np.random.random(shape) * 5 + offset).astype(dtype)
    negative = np.any(array < 0)
    with expected_warnings(["`multichannel` is a deprecated argument"]):
        output = unsharp_mask(array, radius, amount, multichannel=multichannel,
                              preserve_range=preserve)
    if preserve is False:
        assert np.any(output <= 1)
        assert np.any(output >= -1)
        if negative is False:
            assert np.any(output >= 0)
    assert output.dtype in [np.float32, np.float64]
    assert output.shape == shape

    # providing multichannel positionally also raises a warning
    with expected_warnings(["Providing the `multichannel`"]):
        output = unsharp_mask(array, radius, amount, multichannel, preserve)


@parametrize("shape,channel_axis",
             [((16, 16), None),
              ((15, 15, 2), -1),
              ((13, 17, 3), -1)])
@parametrize("preserve", [False, True])
@parametrize("dtype", [np.uint8, np.float16, np.float32, np.float64])
def test_unsharp_masking_dtypes(shape, channel_axis, preserve, dtype):
    radius = 2.0
    amount = 1.0
    array = (np.random.random(shape) * 10).astype(dtype, copy=False)
    negative = np.any(array < 0)
    output = unsharp_mask(array, radius, amount, preserve_range=preserve,
                          channel_axis=channel_axis)
    if preserve is False:
        assert np.any(output <= 1)
        assert np.any(output >= -1)
        if negative is False:
            assert np.any(output >= 0)
    assert output.dtype == _supported_float_type(dtype)
    assert output.shape == shape
