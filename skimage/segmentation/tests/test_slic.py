from itertools import product

import pytest
import numpy as np
from skimage.segmentation import slic

from skimage._shared import testing
from skimage._shared.testing import test_parallel, assert_equal


@test_parallel()
def test_color_2d():
    rnd = np.random.RandomState(0)
    img = np.zeros((20, 21, 3))
    img[:10, :10, 0] = 1
    img[10:, :10, 1] = 1
    img[10:, 10:, 2] = 1
    img += 0.01 * rnd.normal(size=img.shape)
    img[img > 1] = 1
    img[img < 0] = 0
    seg = slic(img, n_segments=4, sigma=0, enforce_connectivity=False,
               start_label=0)

    # we expect 4 segments
    assert_equal(len(np.unique(seg)), 4)
    assert_equal(seg.shape, img.shape[:-1])
    assert_equal(seg[:10, :10], 0)
    assert_equal(seg[10:, :10], 2)
    assert_equal(seg[:10, 10:], 1)
    assert_equal(seg[10:, 10:], 3)


def test_multichannel_2d():
    rnd = np.random.RandomState(0)
    img = np.zeros((20, 20, 8))
    img[:10, :10, 0:2] = 1
    img[:10, 10:, 2:4] = 1
    img[10:, :10, 4:6] = 1
    img[10:, 10:, 6:8] = 1
    img += 0.01 * rnd.normal(size=img.shape)
    img = np.clip(img, 0, 1, out=img)
    seg = slic(img, n_segments=4, enforce_connectivity=False, start_label=0)

    # we expect 4 segments
    assert_equal(len(np.unique(seg)), 4)
    assert_equal(seg.shape, img.shape[:-1])
    assert_equal(seg[:10, :10], 0)
    assert_equal(seg[10:, :10], 2)
    assert_equal(seg[:10, 10:], 1)
    assert_equal(seg[10:, 10:], 3)


def test_gray_2d():
    rnd = np.random.RandomState(0)
    img = np.zeros((20, 21))
    img[:10, :10] = 0.33
    img[10:, :10] = 0.67
    img[10:, 10:] = 1.00
    img += 0.0033 * rnd.normal(size=img.shape)
    img[img > 1] = 1
    img[img < 0] = 0
    seg = slic(img, sigma=0, n_segments=4, compactness=1,
               multichannel=False, convert2lab=False, start_label=0)

    assert_equal(len(np.unique(seg)), 4)
    assert_equal(seg.shape, img.shape)
    assert_equal(seg[:10, :10], 0)
    assert_equal(seg[10:, :10], 2)
    assert_equal(seg[:10, 10:], 1)
    assert_equal(seg[10:, 10:], 3)


def test_color_3d():
    rnd = np.random.RandomState(0)
    img = np.zeros((20, 21, 22, 3))
    slices = []
    for dim_size in img.shape[:-1]:
        midpoint = dim_size // 2
        slices.append((slice(None, midpoint), slice(midpoint, None)))
    slices = list(product(*slices))
    colors = list(product(*(([0, 1],) * 3)))
    for s, c in zip(slices, colors):
        img[s] = c
    img += 0.01 * rnd.normal(size=img.shape)
    img[img > 1] = 1
    img[img < 0] = 0
    seg = slic(img, sigma=0, n_segments=8, start_label=0)

    assert_equal(len(np.unique(seg)), 8)
    for s, c in zip(slices, range(8)):
        assert_equal(seg[s], c)


def test_gray_3d():
    rnd = np.random.RandomState(0)
    img = np.zeros((20, 21, 22))
    slices = []
    for dim_size in img.shape:
        midpoint = dim_size // 2
        slices.append((slice(None, midpoint), slice(midpoint, None)))
    slices = list(product(*slices))
    shades = np.arange(0, 1.000001, 1.0 / 7)
    for s, sh in zip(slices, shades):
        img[s] = sh
    img += 0.001 * rnd.normal(size=img.shape)
    img[img > 1] = 1
    img[img < 0] = 0
    seg = slic(img, sigma=0, n_segments=8, compactness=1,
               multichannel=False, convert2lab=False, start_label=0)

    assert_equal(len(np.unique(seg)), 8)
    for s, c in zip(slices, range(8)):
        assert_equal(seg[s], c)


def test_list_sigma():
    rnd = np.random.RandomState(0)
    img = np.array([[1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1]], np.float)
    img += 0.1 * rnd.normal(size=img.shape)
    result_sigma = np.array([[0, 0, 0, 1, 1, 1],
                             [0, 0, 0, 1, 1, 1]], np.int)
    seg_sigma = slic(img, n_segments=2, sigma=[1, 50, 1],
                     multichannel=False, start_label=0)
    assert_equal(seg_sigma, result_sigma)


def test_spacing():
    rnd = np.random.RandomState(0)
    img = np.array([[1, 1, 1, 0, 0],
                    [1, 1, 0, 0, 0]], np.float)
    result_non_spaced = np.array([[0, 0, 0, 1, 1],
                                  [0, 0, 1, 1, 1]], np.int)
    result_spaced = np.array([[0, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1]], np.int)
    img += 0.1 * rnd.normal(size=img.shape)
    seg_non_spaced = slic(img, n_segments=2, sigma=0, multichannel=False,
                          compactness=1.0, start_label=0)
    seg_spaced = slic(img, n_segments=2, sigma=0, spacing=[1, 500, 1],
                      compactness=1.0, multichannel=False, start_label=0)
    assert_equal(seg_non_spaced, result_non_spaced)
    assert_equal(seg_spaced, result_spaced)


def test_invalid_lab_conversion():
    img = np.array([[1, 1, 1, 0, 0],
                    [1, 1, 0, 0, 0]], np.float) + 1
    with testing.raises(ValueError):
        slic(img, multichannel=True, convert2lab=True, start_label=0)


def test_enforce_connectivity():
    img = np.array([[0, 0, 0, 1, 1, 1],
                    [1, 0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 1, 0]], np.float)

    segments_connected = slic(img, 2, compactness=0.0001,
                              enforce_connectivity=True,
                              convert2lab=False, start_label=0)
    segments_disconnected = slic(img, 2, compactness=0.0001,
                                 enforce_connectivity=False,
                                 convert2lab=False, start_label=0)

    # Make sure nothing fatal occurs (e.g. buffer overflow) at low values of
    # max_size_factor
    segments_connected_low_max = slic(img, 2, compactness=0.0001,
                                      enforce_connectivity=True,
                                      convert2lab=False,
                                      max_size_factor=0.8,
                                      start_label=0)

    result_connected = np.array([[0, 0, 0, 1, 1, 1],
                                 [0, 0, 0, 1, 1, 1],
                                 [0, 0, 0, 1, 1, 1]], np.float)

    result_disconnected = np.array([[0, 0, 0, 1, 1, 1],
                                    [1, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 1, 1, 0]], np.float)

    assert_equal(segments_connected, result_connected)
    assert_equal(segments_disconnected, result_disconnected)
    assert_equal(segments_connected_low_max, result_connected)


def test_slic_zero():
    # Same as test_color_2d but with slic_zero=True
    rnd = np.random.RandomState(0)
    img = np.zeros((20, 21, 3))
    img[:10, :10, 0] = 1
    img[10:, :10, 1] = 1
    img[10:, 10:, 2] = 1
    img += 0.01 * rnd.normal(size=img.shape)
    img[img > 1] = 1
    img[img < 0] = 0
    seg = slic(img, n_segments=4, sigma=0, slic_zero=True, start_label=0)

    # we expect 4 segments
    assert_equal(len(np.unique(seg)), 4)
    assert_equal(seg.shape, img.shape[:-1])
    assert_equal(seg[:10, :10], 0)
    assert_equal(seg[10:, :10], 2)
    assert_equal(seg[:10, 10:], 1)
    assert_equal(seg[10:, 10:], 3)


def test_more_segments_than_pixels():
    rnd = np.random.RandomState(0)
    img = np.zeros((20, 21))
    img[:10, :10] = 0.33
    img[10:, :10] = 0.67
    img[10:, 10:] = 1.00
    img += 0.0033 * rnd.normal(size=img.shape)
    img[img > 1] = 1
    img[img < 0] = 0
    seg = slic(img, sigma=0, n_segments=500, compactness=1,
               multichannel=False, convert2lab=False, start_label=0)
    assert np.all(seg.ravel() == np.arange(seg.size))


def test_color_2d_mask():
    rnd = np.random.RandomState(0)
    msk = np.zeros((20, 21))
    msk[2:-2, 2:-2] = 1
    img = np.zeros((20, 21, 3))
    img[:10, :10, 0] = 1
    img[10:, :10, 1] = 1
    img[10:, 10:, 2] = 1
    img += 0.01 * rnd.normal(size=img.shape)
    np.clip(img, 0, 1, out=img)
    seg = slic(img, n_segments=4, sigma=0, enforce_connectivity=False,
               mask=msk)

    # we expect 4 segments + masked area
    assert_equal(len(np.unique(seg)), 5)
    assert_equal(seg.shape, img.shape[:-1])
    # segments
    assert_equal(seg[2:10, 2:10], 1)
    assert_equal(seg[10:-2, 2:10], 4)
    assert_equal(seg[2:10, 10:-2], 2)
    assert_equal(seg[10:-2, 10:-2], 3)
    # non masked area
    assert_equal(seg[:2, :], 0)
    assert_equal(seg[-2:, :], 0)
    assert_equal(seg[:, :2], 0)
    assert_equal(seg[:, -2:], 0)


def test_multichannel_2d_mask():
    rnd = np.random.RandomState(0)
    msk = np.zeros((20, 20))
    msk[2:-2, 2:-2] = 1
    img = np.zeros((20, 20, 8))
    img[:10, :10, 0:2] = 1
    img[:10, 10:, 2:4] = 1
    img[10:, :10, 4:6] = 1
    img[10:, 10:, 6:8] = 1
    img += 0.01 * rnd.normal(size=img.shape)
    np.clip(img, 0, 1, out=img)
    seg = slic(img, n_segments=4, enforce_connectivity=False,
               mask=msk)

    # we expect 4 segments + masked area
    assert_equal(len(np.unique(seg)), 5)
    assert_equal(seg.shape, img.shape[:-1])
    # segments
    assert_equal(seg[2:10, 2:10], 2)
    assert_equal(seg[2:10, 10:-2], 1)
    assert_equal(seg[10:-2, 2:10], 4)
    assert_equal(seg[10:-2, 10:-2], 3)
    # non masked area
    assert_equal(seg[:2, :], 0)
    assert_equal(seg[-2:, :], 0)
    assert_equal(seg[:, :2], 0)
    assert_equal(seg[:, -2:], 0)


def test_gray_2d_mask():
    rnd = np.random.RandomState(0)
    msk = np.zeros((20, 21))
    msk[2:-2, 2:-2] = 1
    img = np.zeros((20, 21))
    img[:10, :10] = 0.33
    img[10:, :10] = 0.67
    img[10:, 10:] = 1.00
    img += 0.0033 * rnd.normal(size=img.shape)
    np.clip(img, 0, 1, out=img)
    seg = slic(img, sigma=0, n_segments=4, compactness=1,
               multichannel=False, convert2lab=False, mask=msk)

    assert_equal(len(np.unique(seg)), 5)
    assert_equal(seg.shape, img.shape)
    # segments
    assert_equal(seg[2:10, 2:10], 1)
    assert_equal(seg[2:10, 10:-2], 2)
    assert_equal(seg[10:-2, 2:10], 3)
    assert_equal(seg[10:-2, 10:-2], 4)
    # non masked area
    assert_equal(seg[:2, :], 0)
    assert_equal(seg[-2:, :], 0)
    assert_equal(seg[:, :2], 0)
    assert_equal(seg[:, -2:], 0)


def test_list_sigma_mask():
    rnd = np.random.RandomState(0)
    msk = np.zeros((2, 6))
    msk[:, 1:-1] = 1
    img = np.array([[1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1]], np.float)
    img += 0.1 * rnd.normal(size=img.shape)
    result_sigma = np.array([[0, 1, 1, 2, 2, 0],
                             [0, 1, 1, 2, 2, 0]], np.int)
    seg_sigma = slic(img, n_segments=2, sigma=[1, 50, 1],
                     multichannel=False, mask=msk)
    assert_equal(seg_sigma, result_sigma)


def test_spacing_mask():
    rnd = np.random.RandomState(0)
    msk = np.zeros((2, 5))
    msk[:, 1:-1] = 1
    img = np.array([[1, 1, 1, 0, 0],
                    [1, 1, 0, 0, 0]], np.float)
    result_non_spaced = np.array([[0, 1, 1, 2, 0],
                                  [0, 1, 2, 2, 0]], np.int)
    result_spaced = np.array([[0, 1, 1, 1, 0],
                              [0, 2, 2, 2, 0]], np.int)
    img += 0.1 * rnd.normal(size=img.shape)
    seg_non_spaced = slic(img, n_segments=2, sigma=0, multichannel=False,
                          compactness=1.0, mask=msk)
    seg_spaced = slic(img, n_segments=2, sigma=0, spacing=[1, 50, 1],
                      compactness=1.0, multichannel=False, mask=msk)
    assert_equal(seg_non_spaced, result_non_spaced)
    assert_equal(seg_spaced, result_spaced)


def test_enforce_connectivity_mask():
    msk = np.zeros((3, 6))
    msk[:, 1:-1] = 1
    img = np.array([[0, 0, 0, 1, 1, 1],
                    [1, 0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 1, 0]], np.float)

    segments_connected = slic(img, 2, compactness=0.0001,
                              enforce_connectivity=True,
                              convert2lab=False, mask=msk)
    segments_disconnected = slic(img, 2, compactness=0.0001,
                                 enforce_connectivity=False,
                                 convert2lab=False, mask=msk)

    # Make sure nothing fatal occurs (e.g. buffer overflow) at low values of
    # max_size_factor
    segments_connected_low_max = slic(img, 2, compactness=0.0001,
                                      enforce_connectivity=True,
                                      convert2lab=False,
                                      max_size_factor=0.8, mask=msk)

    result_connected = np.array([[0, 1, 1, 2, 2, 0],
                                 [0, 1, 1, 2, 2, 0],
                                 [0, 1, 1, 2, 2, 0]], np.float)

    result_disconnected = np.array([[0, 1, 1, 2, 2, 0],
                                    [0, 1, 1, 2, 2, 0],
                                    [0, 1, 1, 2, 2, 0]], np.float)

    assert_equal(segments_connected, result_connected)
    assert_equal(segments_disconnected, result_disconnected)
    assert_equal(segments_connected_low_max, result_connected)


def test_slic_zero_mask():

    rnd = np.random.RandomState(0)
    msk = np.zeros((20, 21))
    msk[2:-2, 2:-2] = 1
    img = np.zeros((20, 21, 3))
    img[:10, :10, 0] = 1
    img[10:, :10, 1] = 1
    img[10:, 10:, 2] = 1
    img += 0.01 * rnd.normal(size=img.shape)
    np.clip(img, 0, 1, out=img)
    seg = slic(img, n_segments=4, sigma=0, slic_zero=True,
               mask=msk)

    # we expect 4 segments + masked area
    assert_equal(len(np.unique(seg)), 5)
    assert_equal(seg.shape, img.shape[:-1])
    # segments
    assert_equal(seg[2:10, 2:10], 1)
    assert_equal(seg[2:10, 10:-2], 2)
    assert_equal(seg[10:-2, 2:10], 3)
    assert_equal(seg[10:-2, 10:-2], 4)
    # non masked area
    assert_equal(seg[:2, :], 0)
    assert_equal(seg[-2:, :], 0)
    assert_equal(seg[:, :2], 0)
    assert_equal(seg[:, -2:], 0)


def test_more_segments_than_pixels_mask():
    rnd = np.random.RandomState(0)
    msk = np.zeros((20, 21))
    msk[2:-2, 2:-2] = 1
    img = np.zeros((20, 21))
    img[:10, :10] = 0.33
    img[10:, :10] = 0.67
    img[10:, 10:] = 1.00
    img += 0.0033 * rnd.normal(size=img.shape)
    np.clip(img, 0, 1, out=img)
    seg = slic(img, sigma=0, n_segments=500, compactness=1,
               multichannel=False, convert2lab=False, mask=msk)

    expected = np.arange(seg[2:-2, 2:-2].size) + 1
    assert np.all(seg[2:-2, 2:-2].ravel() == expected)


def test_color_3d_mask():

    msk = np.zeros((20, 21, 22))
    msk[2:-2, 2:-2, 2:-2] = 1

    rnd = np.random.RandomState(0)
    img = np.zeros((20, 21, 22, 3))
    slices = []
    for dim_size in msk.shape:
        midpoint = dim_size // 2
        slices.append((slice(None, midpoint), slice(midpoint, None)))
    slices = list(product(*slices))
    colors = list(product(*(([0, 1],) * 3)))
    for s, c in zip(slices, colors):
        img[s] = c
    img += 0.01 * rnd.normal(size=img.shape)
    np.clip(img, 0, 1, out=img)

    seg = slic(img, sigma=0, n_segments=8, mask=msk)

    # we expect 8 segments + masked area
    assert_equal(len(np.unique(seg)), 9)
    for s, c in zip(slices, range(1, 9)):
        assert_equal(seg[s][2:-2, 2:-2, 2:-2], c)


def test_gray_3d_mask():

    msk = np.zeros((20, 21, 22))
    msk[2:-2, 2:-2, 2:-2] = 1

    rnd = np.random.RandomState(0)
    img = np.zeros((20, 21, 22))
    slices = []
    for dim_size in img.shape:
        midpoint = dim_size // 2
        slices.append((slice(None, midpoint), slice(midpoint, None)))
    slices = list(product(*slices))
    shades = np.linspace(0, 1, 8)
    for s, sh in zip(slices, shades):
        img[s] = sh
    img += 0.001 * rnd.normal(size=img.shape)
    np.clip(img, 0, 1, out=img)
    seg = slic(img, sigma=0, n_segments=8, multichannel=False,
               convert2lab=False, mask=msk)

    # we expect 8 segments + masked area
    assert_equal(len(np.unique(seg)), 9)
    for s, c in zip(slices, range(1, 9)):
        assert_equal(seg[s][2:-2, 2:-2, 2:-2], c)


@pytest.mark.parametrize("dtype", ['float32', 'float64', 'uint8', 'int'])
def test_dtype_support(dtype):
    img = np.random.rand(28, 28).astype(dtype)

    # Simply run the function to assert that it runs without error
    slic(img, start_label=1)
