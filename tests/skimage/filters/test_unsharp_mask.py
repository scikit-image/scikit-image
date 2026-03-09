import warnings

import numpy as np
import pytest

from skimage._shared.utils import _supported_float_type
from skimage.filters import unsharp_mask


@pytest.mark.parametrize(
    "shape,multichannel",
    [
        ((29,), False),
        ((40, 4), True),
        ((32, 32), False),
        ((29, 31, 3), True),
        ((13, 17, 4, 8), False),
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
    radius, amount, shape, multichannel, dtype, offset, preserve
):
    array = np.random.random(shape)
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
    "shape,multichannel", [((32, 32), False), ((15, 15, 2), True), ((17, 19, 3), True)]
)
@pytest.mark.parametrize("radius", [(0.0, 0.0), (1.0, 1.0), (2.0, 1.5)])
@pytest.mark.parametrize("preserve", [False, True])
def test_unsharp_masking_with_different_radii(radius, shape, multichannel, preserve):
    amount = 1.0
    dtype = np.float64
    array = (np.random.random(shape) * 96).astype(dtype)
    if preserve is False:
        array /= max(np.abs(array).max(), 1.0)
    channel_axis = -1 if multichannel else None
    output = unsharp_mask(
        array, radius, amount, preserve_range=preserve, channel_axis=channel_axis
    )
    assert output.dtype in [np.float32, np.float64]
    assert output.shape == shape


@pytest.mark.parametrize(
    "shape,channel_axis",
    [
        ((16, 16), None),
        ((15, 15, 2), -1),
        ((13, 17, 3), -1),
        ((2, 15, 15), 0),
        ((3, 13, 17), 0),
    ],
)
@pytest.mark.parametrize("offset", [-5, 0, 5])
@pytest.mark.parametrize("preserve", [False, True])
def test_unsharp_masking_with_different_ranges(shape, offset, channel_axis, preserve):
    radius = 2.0
    amount = 1.0
    dtype = np.int16
    array = (np.random.random(shape) * 5 + offset).astype(dtype)
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
    "shape,channel_axis", [((16, 16), None), ((15, 15, 2), -1), ((13, 17, 3), -1)]
)
@pytest.mark.parametrize("offset", [-5, 0, 5])
@pytest.mark.parametrize("preserve", [False, True])
def test_unsharp_masking_with_different_ranges_deprecated(
    shape, offset, channel_axis, preserve
):
    radius = 2.0
    amount = 1.0
    dtype = np.int16
    array = (np.random.random(shape) * 5 + offset).astype(dtype)
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
    "shape,channel_axis", [((16, 16), None), ((15, 15, 2), -1), ((13, 17, 3), -1)]
)
@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("dtype", [np.uint8, np.float16, np.float32, np.float64])
def test_unsharp_masking_dtypes(shape, channel_axis, preserve, dtype):
    radius = 2.0
    amount = 1.0
    array = (np.random.random(shape) * 10).astype(dtype, copy=False)
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


def test_unsharp_mask_channel_axis_negative():
    """Test that channel_axis=-1 works correctly for RGB images.

    Regression test for https://github.com/scikit-image/scikit-image/issues/7264
    """
    # Create a simple RGB image with non-zero values
    rgb_image = np.random.rand(32, 32, 3).astype(np.float32)

    # Apply unsharp mask with channel_axis=-1
    result = unsharp_mask(rgb_image, channel_axis=-1)

    # Verify the result is not all zeros (the bug caused black output)
    assert not np.all(result == 0), "Output should not be all zeros"

    # Verify output has values in expected range
    assert np.any(result > 0), "Output should have positive values"
    assert result.min() >= 0, "Output should be non-negative for non-negative input"
    assert result.max() <= 1, "Output should be at most 1 for normalized input"

    # Also test with uint8 image (the common case from the issue)
    rgb_uint8 = (rgb_image * 255).astype(np.uint8)
    result_uint8 = unsharp_mask(rgb_uint8, channel_axis=-1)

    assert not np.all(result_uint8 == 0), "Output should not be all zeros for uint8 input"
    assert np.any(result_uint8 > 0), "Output should have positive values for uint8 input"
