import pytest
import sys
import os
from skimage.exposure import hdr
from skimage import exposure
import numpy as np
# sys.path.insert(0, os.path.abspath('../'))  # noqa: E402
# import hdr

from skimage._shared.testing import assert_array_almost_equal, \
    assert_almost_equal, expected_warnings  # noqa: E402


# Make test image
depth = 10
h_bit = 2**depth - 1

x = np.arange(-1, 1, 0.5)
xx, yy = np.meshgrid(x, x, sparse=True)

z = (np.cos(xx**2 + yy**2)) ** 2 + 0.1


z_norm = z / np.max(z[:])

z = np.int32(z_norm * h_bit)

im1 = z.copy()
im1[im1 > h_bit] = h_bit

im2 = z // 2
im2[im2 <= 0] = 1
im2[im2 > h_bit] = h_bit

im3 = z // 3
im3[im3 <= 0] = 1
im3[im3 > h_bit] = h_bit

images = np.stack((im1, im2, im3), axis=0)
images_rgb = np.broadcast_to(images[..., None], images.shape + (3,))
exposure = [1 / 50, 1 / 100, 1 / 150]


def test_make_hdr_grey():
    # when
    radiance_map = hdr.get_crf(
        images, exposure, depth=depth, channel_axis=None)
    hdr_im = hdr.make_hdr(images, exposure, radiance_map,
                          depth=depth, channel_axis=None)
    hdr_norm = hdr_im / np.nanmax(hdr_im[:])
    # then
    assert_array_almost_equal(hdr_norm, z_norm, decimal=2)


def test_make_hdr_rgb():
    # when
    radiance_map = hdr.get_crf(images_rgb, exposure,
                               depth=depth, channel_axis=3)
    hdr_im = hdr.make_hdr(images_rgb, exposure, radiance_map,
                          depth=depth, channel_axis=3)
    hdr_norm = hdr_im / np.nanmax(hdr_im[:])
    # then
    expected_rgb = np.broadcast_to(z_norm[..., None], z_norm.shape + (3,))

    assert_array_almost_equal(hdr_norm, expected_rgb, decimal=2)


image_float = np.ones([3, 10, 10, 3], dtype=np.float64)
image_dim = np.ones([3, 10, 10, 3, 3], dtype=np.int32)
rad_map = [0.1, 0.2, 0.3]


def test_float_crf():
    """Test that floats are not accepted as images in `get_crf` """
    with pytest.raises(ValueError):
        hdr.get_crf(image_float, exposure)


def test_float_hdr():
    """Test that floats are not accepted as images in `make_hdr` """
    with pytest.raises(ValueError):
        hdr.make_hdr(image_float, exposure, rad_map)


def test_to_many_dim_crf():
    """Test that images with to many dimensions are not accepted `get_crf` """
    with pytest.raises(ValueError):
        hdr.get_crf(image_dim, exposure, channel_axis=3)


def test_to_many_dim_hdr():
    """Test that images with to many dimensions are not accepted `make_hdr` """
    with pytest.raises(ValueError):
        hdr.make_hdr(image_dim, exposure, rad_map, channel_axis=3)
# class TestMatchHistogram:

    # image_rgb = data.chelsea()
