import warnings

import numpy as np
import pytest

from _skimage2._shared.utils import _supported_float_type
from skimage.filters import unsharp_mask


@pytest.mark.parametrize(
    "shape,multichannel,seed",
    [
        ((29,), False, 2691658343),
        ((40, 4), True, 961080854),
        ((32, 32), False, 859949972),
        ((29, 31, 3), True, 1195250906),
        ((13, 17, 4, 8), False, 1909137959),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.uint32,
        np.int32,
        np.uint64,
        np.int64,
        np.float32,
        np.float64,
    ],
)
@pytest.mark.parametrize("radius", [0, 0.1, 2.0])
@pytest.mark.parametrize("amount", [0.0, 0.5, 2.0, -1.0])
@pytest.mark.parametrize("offset", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("preserve", [False, True])
def test_unsharp_masking_output_type_and_shape(
    radius, amount, shape, multichannel, seed, dtype, offset, preserve
):
    array = np.random.RandomState(seed).random(shape)
    array = (array + offset) * 128
    with warnings.catch_warnings():
        # Ignore arch specific warning on arm64, armhf, ppc64el, riscv64, s390x
        # https://github.com/scikit-image/scikit-image/issues/7391
        warnings.filterwarnings(
            action="ignore",
            category=RuntimeWarning,
            message="invalid value encountered in cast",
        )
        array = array.astype(dtype)

    if (preserve is False) and (dtype in [np.float32, np.float64]):
        array /= max(np.abs(array).max(), 1.0)
    channel_axis = -1 if multichannel else None
    output = unsharp_mask(
        array, radius, amount, preserve_range=preserve, channel_axis=channel_axis
    )
    assert output.dtype in [np.float32, np.float64]
    assert output.shape == shape


@pytest.mark.parametrize(
    "shape,multichannel,seed",
    [
        ((32, 32), False, 239302397),
        ((15, 15, 2), True, 2870300792),
        ((17, 19, 3), True, 3028283166),
    ],
)
@pytest.mark.parametrize("radius", [(0.0, 0.0), (1.0, 1.0), (2.0, 1.5)])
@pytest.mark.parametrize("preserve", [False, True])
def test_unsharp_masking_with_different_radii(
    radius, shape, multichannel, seed, preserve
):
    amount = 1.0
    dtype = np.float64
    array = (np.random.RandomState(seed).random(shape) * 96).astype(dtype)
    if preserve is False:
        array /= max(np.abs(array).max(), 1.0)
    channel_axis = -1 if multichannel else None
    output = unsharp_mask(
        array, radius, amount, preserve_range=preserve, channel_axis=channel_axis
    )
    assert output.dtype in [np.float32, np.float64]
    assert output.shape == shape


@pytest.mark.parametrize(
    "shape,channel_axis,seed",
    [
        ((16, 16), None, 3896196842),
        ((15, 15, 2), -1, 2349136640),
        ((13, 17, 3), -1, 864018234),
        ((2, 15, 15), 0, 1183432730),
        ((3, 13, 17), 0, 4176181874),
    ],
)
@pytest.mark.parametrize("offset", [-5, 0, 5])
@pytest.mark.parametrize("preserve", [False, True])
def test_unsharp_masking_with_different_ranges(
    shape, offset, channel_axis, preserve, seed
):
    radius = 2.0
    amount = 1.0
    dtype = np.int16
    array = (np.random.RandomState(seed).random(shape) * 5 + offset).astype(dtype)
    negative = np.any(array < 0)
    output = unsharp_mask(
        array, radius, amount, preserve_range=preserve, channel_axis=channel_axis
    )
    if preserve is False:
        assert np.any(output <= 1)
        assert np.any(output >= -1)
        if negative is False:
            assert np.any(output >= 0)
    assert output.dtype in [np.float32, np.float64]
    assert output.shape == shape


@pytest.mark.parametrize(
    "shape,channel_axis,seed",
    [
        ((16, 16), None, 3606780155),
        ((15, 15, 2), -1, 3779868710),
        ((13, 17, 3), -1, 2857193517),
    ],
)
@pytest.mark.parametrize("offset", [-5, 0, 5])
@pytest.mark.parametrize("preserve", [False, True])
def test_unsharp_masking_with_different_ranges_deprecated(
    shape, offset, channel_axis, preserve, seed
):
    radius = 2.0
    amount = 1.0
    dtype = np.int16
    array = (np.random.RandomState(seed).random(shape) * 5 + offset).astype(dtype)
    negative = np.any(array < 0)
    output = unsharp_mask(
        array, radius, amount, channel_axis=channel_axis, preserve_range=preserve
    )
    if preserve is False:
        assert np.any(output <= 1)
        assert np.any(output >= -1)
        if negative is False:
            assert np.any(output >= 0)
    assert output.dtype in [np.float32, np.float64]
    assert output.shape == shape


@pytest.mark.parametrize(
    "shape,channel_axis,seed",
    [
        ((16, 16), None, 3378941598),
        ((15, 15, 2), -1, 2524096514),
        ((13, 17, 3), -1, 1078129855),
    ],
)
@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32, np.float64])
def test_unsharp_masking_dtypes(shape, channel_axis, seed, preserve, dtype):
    radius = 2.0
    amount = 1.0
    array = (np.random.RandomState(seed).random(shape) * 10).astype(dtype, copy=False)
    negative = np.any(array < 0)
    output = unsharp_mask(
        array, radius, amount, preserve_range=preserve, channel_axis=channel_axis
    )
    if preserve is False:
        assert np.any(output <= 1)
        assert np.any(output >= -1)
        if negative is False:
            assert np.any(output >= 0)
    assert output.dtype == _supported_float_type(dtype)
    assert output.shape == shape
