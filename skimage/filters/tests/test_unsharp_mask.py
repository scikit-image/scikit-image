import numpy as np
from skimage.filters import unsharp_mask
from skimage._shared.testing import parametrize


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
    output = unsharp_mask(array, radius, amount, multichannel, preserve)
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
    output = unsharp_mask(array, radius, amount, multichannel, preserve)
    assert output.dtype in [np.float32, np.float64]
    assert output.shape == shape


@parametrize("shape,multichannel",
             [((16, 16), False),
              ((15, 15, 2), True),
              ((13, 17, 3), True)])
@parametrize("offset", [-5, 0, 5])
@parametrize("preserve", [False, True])
def test_unsharp_masking_with_different_ranges(shape, offset,
                                               multichannel, preserve):
    radius = 2.0
    amount = 1.0
    dtype = np.int16
    array = (np.random.random(shape) * 5 + offset).astype(dtype)
    negative = np.any(array < 0)
    output = unsharp_mask(array, radius, amount, multichannel, preserve)
    if preserve is False:
        assert np.any(output <= 1)
        assert np.any(output >= -1)
        if negative is False:
            assert np.any(output >= 0)
    assert output.dtype in [np.float32, np.float64]
    assert output.shape == shape
