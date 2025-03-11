import numpy as np
from numpy.testing import assert_array_equal
import pytest

from skimage import color, data, morphology
from skimage.morphology import isotropic
from skimage.util import img_as_bool

img = color.rgb2gray(data.astronaut())
bw_img = img > 100 / 255.0


def test_non_square_image():
    isotropic_res = isotropic.isotropic_erosion(bw_img[:100, :200], 3)
    binary_res = img_as_bool(morphology.erosion(bw_img[:100, :200], morphology.disk(3)))
    assert_array_equal(isotropic_res, binary_res)


def test_isotropic_erosion():
    isotropic_res = isotropic.isotropic_erosion(bw_img, 3)
    binary_res = img_as_bool(morphology.erosion(bw_img, morphology.disk(3)))
    assert_array_equal(isotropic_res, binary_res)


def _disk_with_spacing(radius, dtype=np.uint8, *, strict_radius=True, spacing=None):
    # Identical to morphology.disk, but with a spacing parameter and without decomposition.
    # This is different from morphology.ellipse which produces a slightly different footprint.
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)

    if spacing is not None:
        X *= spacing[1]
        Y *= spacing[0]

    if not strict_radius:
        radius += 0.5
    return np.array((X**2 + Y**2) <= radius**2, dtype=dtype)


def test_isotropic_erosion_spacing():
    isotropic_res = isotropic.isotropic_dilation(bw_img, 6, spacing=(1, 2))
    binary_res = img_as_bool(
        morphology.dilation(bw_img, _disk_with_spacing(6, spacing=(1, 2)))
    )
    assert_array_equal(isotropic_res, binary_res)


def test_isotropic_dilation():
    isotropic_res = isotropic.isotropic_dilation(bw_img, 3)
    binary_res = img_as_bool(morphology.dilation(bw_img, morphology.disk(3)))
    assert_array_equal(isotropic_res, binary_res)


def test_isotropic_closing():
    isotropic_res = isotropic.isotropic_closing(bw_img, 3)
    binary_res = img_as_bool(morphology.closing(bw_img, morphology.disk(3)))
    assert_array_equal(isotropic_res, binary_res)


def test_isotropic_opening():
    isotropic_res = isotropic.isotropic_opening(bw_img, 3)
    binary_res = img_as_bool(morphology.opening(bw_img, morphology.disk(3)))
    assert_array_equal(isotropic_res, binary_res)


def test_footprint_overflow():
    img = np.zeros((20, 20), dtype=bool)
    img[2:19, 2:19] = True
    isotropic_res = isotropic.isotropic_erosion(img, 9)
    binary_res = img_as_bool(morphology.erosion(img, morphology.disk(9)))
    assert_array_equal(isotropic_res, binary_res)


def test_out_argument():
    for func in (isotropic.isotropic_erosion, isotropic.isotropic_dilation):
        radius = 3
        img = np.ones((10, 10))
        out = np.zeros_like(img)
        out_saved = out.copy()
        func(img, radius, out=out)
        assert np.any(out != out_saved)
        assert_array_equal(out, func(img, radius))


@pytest.mark.parametrize(
    "func",
    [
        isotropic.isotropic_erosion,
        isotropic.isotropic_dilation,
        isotropic.isotropic_closing,
        isotropic.isotropic_opening,
    ],
)
def test_isotrophic_errors(func):
    with pytest.raises(TypeError, match="Input image must be a binary image"):
        func(img, radius=2)
