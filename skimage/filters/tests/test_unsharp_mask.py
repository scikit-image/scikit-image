import numpy as np
import pytest

from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.filters import unsharp_mask


@pytest.mark.parametrize("shape,multichannel",
                         [((29,), False),
                          ((40, 4), True),
                          ((32, 32), False),
                          ((29, 31, 3), True),
                          ((13, 17, 4, 8), False)])
@pytest.mark.parametrize("dtype", [np.uint8, np.int8,
                                   np.uint16, np.int16,
                                   np.uint32, np.int32,
                                   np.uint64, np.int64,
                                   np.float32, np.float64])
@pytest.mark.parametrize("radius", [0, 0.1, 2.0])
@pytest.mark.parametrize("amount", [0.0, 0.5, 2.0, -1.0])
@pytest.mark.parametrize("offset", [-1.0, 0.0, 1.0])
def test_unsharp_masking_output_type_and_shape(
        radius, amount, shape, multichannel, dtype, offset):
    array = np.random.random(shape)
    array = ((array + offset) * 128).astype(dtype)
    channel_axis = -1 if multichannel else None
    output = unsharp_mask(array, radius, amount, channel_axis=channel_axis)
    assert output.dtype in [np.float32, np.float64]
    assert output.shape == shape


@pytest.mark.parametrize("shape,multichannel",
                         [((32, 32), False),
                          ((15, 15, 2), True),
                          ((17, 19, 3), True)])
@pytest.mark.parametrize("radius", [(0.0, 0.0), (1.0, 1.0), (2.0, 1.5)])
def test_unsharp_masking_with_different_radii(radius, shape, multichannel):
    amount = 1.0
    dtype = np.float64
    array = (np.random.random(shape) * 96).astype(dtype)
    channel_axis = -1 if multichannel else None
    output = unsharp_mask(array, radius, amount, channel_axis=channel_axis)
    assert output.dtype in [np.float32, np.float64]
    assert output.shape == shape


@pytest.mark.parametrize("shape,channel_axis",
                         [((16, 16), None),
                          ((15, 15, 2), -1),
                          ((13, 17, 3), -1),
                          ((2, 15, 15), 0),
                          ((3, 13, 17), 0)])
@pytest.mark.parametrize("offset", [-5, 0, 5])
def test_unsharp_masking_with_different_ranges(shape, offset, channel_axis):
    radius = 2.0
    amount = 1.0
    dtype = np.int16
    array = (np.random.random(shape) * 5 + offset).astype(dtype)
    output = unsharp_mask(array, radius, amount, channel_axis=channel_axis)
    assert output.dtype in [np.float32, np.float64]
    assert output.shape == shape


@pytest.mark.parametrize("shape,multichannel",
                         [((16, 16), False),
                          ((15, 15, 2), True),
                          ((13, 17, 3), True)])
@pytest.mark.parametrize("offset", [-5, 0, 5])
def test_unsharp_masking_with_different_ranges_deprecated(shape, offset,
                                                          multichannel):
    radius = 2.0
    amount = 1.0
    dtype = np.int16
    array = (np.random.random(shape) * 5 + offset).astype(dtype)
    with expected_warnings(["`multichannel` is a deprecated argument"]):
        output = unsharp_mask(array, radius, amount, multichannel=multichannel)
    assert output.dtype in [np.float32, np.float64]
    assert output.shape == shape

    # providing multichannel positionally also raises a warning
    with expected_warnings(["Providing the `multichannel`"]):
        output = unsharp_mask(array, radius, amount, multichannel)


@pytest.mark.parametrize("shape,channel_axis",
                         [((16, 16), None),
                          ((15, 15, 2), -1),
                          ((13, 17, 3), -1)])
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32, np.float64])
def test_unsharp_masking_dtypes(shape, channel_axis, dtype):
    radius = 2.0
    amount = 1.0
    array = (np.random.random(shape) * 10).astype(dtype, copy=False)
    output = unsharp_mask(array, radius, amount, channel_axis=channel_axis)
    assert output.dtype == _supported_float_type(dtype)
    assert output.shape == shape
