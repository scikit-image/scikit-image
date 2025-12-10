import numpy as np
import pytest
from skimage.util import crop, bounding_box_crop
from skimage._shared.testing import assert_array_equal, assert_equal


def test_multi_crop():
    arr = np.arange(45).reshape(9, 5)
    out = crop(arr, ((1, 2), (2, 1)))
    assert_array_equal(out[0], [7, 8])
    assert_array_equal(out[-1], [32, 33])
    assert_equal(out.shape, (6, 2))


def test_pair_crop():
    arr = np.arange(45).reshape(9, 5)
    out = crop(arr, (1, 2))
    assert_array_equal(out[0], [6, 7])
    assert_array_equal(out[-1], [31, 32])
    assert_equal(out.shape, (6, 2))


def test_pair_tuple_crop():
    arr = np.arange(45).reshape(9, 5)
    out = crop(arr, ((1, 2),))
    assert_array_equal(out[0], [6, 7])
    assert_array_equal(out[-1], [31, 32])
    assert_equal(out.shape, (6, 2))


def test_int_crop():
    arr = np.arange(45).reshape(9, 5)
    out = crop(arr, 1)
    assert_array_equal(out[0], [6, 7, 8])
    assert_array_equal(out[-1], [36, 37, 38])
    assert_equal(out.shape, (7, 3))


def test_int_tuple_crop():
    arr = np.arange(45).reshape(9, 5)
    out = crop(arr, (1,))
    assert_array_equal(out[0], [6, 7, 8])
    assert_array_equal(out[-1], [36, 37, 38])
    assert_equal(out.shape, (7, 3))


def test_copy_crop():
    arr = np.arange(45).reshape(9, 5)
    out0 = crop(arr, 1, copy=True)
    assert out0.flags.c_contiguous
    out0[0, 0] = 100
    assert not np.any(arr == 100)
    assert not np.may_share_memory(arr, out0)

    out1 = crop(arr, 1)
    out1[0, 0] = 100
    assert arr[1, 1] == 100
    assert np.may_share_memory(arr, out1)


def test_zero_crop():
    arr = np.arange(45).reshape(9, 5)
    out = crop(arr, 0)
    assert out.shape == (9, 5)


def test_np_int_crop():
    arr = np.arange(45).reshape(9, 5)
    out1 = crop(arr, np.int64(1))
    out2 = crop(arr, np.int32(1))
    assert_array_equal(out1, out2)
    assert out1.shape == (7, 3)


def test_bounding_box_basic_2d():
    arr = np.arange(6 * 7).reshape(6, 7)
    out = bounding_box_crop(arr, ((1, 2), (5, 6)))
    expected = arr[1:5, 2:6]
    assert_array_equal(out, expected)
    assert_equal(out.shape, (4, 4))


def test_bounding_box_float_semantics():
    arr = np.arange(6 * 7).reshape(6, 7)
    out = bounding_box_crop(arr, ((1.2, 2.1), (4.0, 5.9)))
    expected = arr[1:4, 2:6]  # floor mins, ceil maxes, stop is exclusive
    assert_array_equal(out, expected)
    assert_equal(out.shape, (3, 4))


def test_bounding_box_channel_axis_last():
    rgb = np.arange(32 * 48 * 3).reshape(32, 48, 3)
    out = bounding_box_crop(rgb, ((4, 5), (10, 12)), channel_axis=-1)
    assert_equal(out.shape, (6, 7, 3))
    # Check a couple of values to ensure spatial crop only
    assert_array_equal(out[0, 0], rgb[4, 5])
    assert_array_equal(out[-1, -1], rgb[9, 11])


def test_bounding_box_channel_axis_first():
    chw = np.arange(3 * 32 * 48).reshape(3, 32, 48)
    out = bounding_box_crop(chw, ((4, 5), (10, 12)), channel_axis=0)
    assert_equal(out.shape, (3, 6, 7))
    # Check spatial indices unaffected for the channel axis
    assert_array_equal(out[:, 0, 0], chw[:, 4, 5])
    assert_array_equal(out[:, -1, -1], chw[:, 9, 11])


def test_bounding_box_clip_behavior():
    arr = np.arange(5 * 5).reshape(5, 5)
    # Partially out of bounds: clamp to [0:3] x [0:5]
    out = bounding_box_crop(arr, ((-2, -1), (3, 7)), clip=True)
    assert_equal(out.shape, (3, 5))
    assert_array_equal(out, arr[0:3, 0:5])

    # With clip=False, the same bbox should raise
    with pytest.raises(ValueError):
        _ = bounding_box_crop(arr, ((-2, -1), (3, 7)), clip=False)


def test_bounding_box_degenerate_zero_length():
    arr = np.arange(6 * 7).reshape(6, 7)
    out = bounding_box_crop(arr, ((2, 3), (2, 5)))  # zero length on axis 0
    assert_equal(out.shape, (0, 2))
    # Zero-length slices compare equal trivially
    assert_array_equal(out, arr[2:2, 3:5])


def test_bounding_box_copy_and_order_semantics():
    arr = np.arange(10 * 12).reshape(10, 12)
    out_view = bounding_box_crop(arr, ((1, 2), (9, 11)), copy=False)
    assert np.may_share_memory(arr, out_view)

    out_c = bounding_box_crop(arr, ((1, 2), (9, 11)), copy=True, order='C')
    assert not np.may_share_memory(arr, out_c)
    assert out_c.flags.c_contiguous

    out_f = bounding_box_crop(arr, ((1, 2), (9, 11)), copy=True, order='F')
    assert not np.may_share_memory(arr, out_f)
    assert out_f.flags.f_contiguous


def test_bounding_box_dimensionality_mismatch():
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    # With channel_axis=-1, spatial D=2 but we provide 3D bbox
    with pytest.raises(ValueError):
        _ = bounding_box_crop(arr, ((0, 0, 0), (5, 5, 2)), channel_axis=-1)
