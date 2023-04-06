import itertools
import warnings
import re

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.ndimage import fourier_shift
import scipy.fft as fft

from skimage import img_as_float
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.data import camera, binary_blobs, eagle
from skimage.registration._phase_cross_correlation import (
    phase_cross_correlation, _upsampled_dft
)


@pytest.mark.parametrize('normalization', [None, 'phase'])
def test_correlation(normalization):
    reference_image = fft.fftn(camera())
    shift = (-7, 12)
    shifted_image = fourier_shift(reference_image, shift)

    # pixel precision
    result, _, _ = phase_cross_correlation(reference_image,
                                           shifted_image,
                                           space="fourier",
                                           normalization=normalization)
    assert_allclose(result[:2], -np.array(shift))


@pytest.mark.parametrize('normalization', ['nonexisting'])
def test_correlation_invalid_normalization(normalization):
    reference_image = fft.fftn(camera())
    shift = (-7, 12)
    shifted_image = fourier_shift(reference_image, shift)

    # pixel precision
    with pytest.raises(ValueError):
        phase_cross_correlation(reference_image,
                                shifted_image,
                                space="fourier",
                                normalization=normalization)


@pytest.mark.parametrize('normalization', [None, 'phase'])
def test_subpixel_precision(normalization):
    reference_image = fft.fftn(camera())
    subpixel_shift = (-2.4, 1.32)
    shifted_image = fourier_shift(reference_image, subpixel_shift)

    # subpixel precision
    result, _, _ = phase_cross_correlation(reference_image,
                                           shifted_image,
                                           upsample_factor=100,
                                           space="fourier",
                                           normalization=normalization)
    assert_allclose(result[:2], -np.array(subpixel_shift), atol=0.05)


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_real_input(dtype):
    reference_image = camera().astype(dtype, copy=False)
    subpixel_shift = (-2.4, 1.32)
    shifted_image = fourier_shift(fft.fftn(reference_image), subpixel_shift)
    shifted_image = fft.ifftn(shifted_image).real.astype(dtype, copy=False)

    # subpixel precision
    result, error, diffphase = phase_cross_correlation(reference_image,
                                                       shifted_image,
                                                       upsample_factor=100)
    assert result.dtype == _supported_float_type(dtype)
    assert_allclose(result[:2], -np.array(subpixel_shift), atol=0.05)


def test_size_one_dimension_input():
    # take a strip of the input image
    reference_image = fft.fftn(camera()[:, 15]).reshape((-1, 1))
    subpixel_shift = (-2.4, 4)
    shifted_image = fourier_shift(reference_image, subpixel_shift)

    # subpixel precision
    result, error, diffphase = phase_cross_correlation(reference_image,
                                                       shifted_image,
                                                       upsample_factor=20,
                                                       space="fourier")
    assert_allclose(result[:2], -np.array((-2.4, 0)), atol=0.05)


def test_3d_input():
    phantom = img_as_float(binary_blobs(length=32, n_dim=3))
    reference_image = fft.fftn(phantom)
    shift = (-2., 1., 5.)
    shifted_image = fourier_shift(reference_image, shift)

    result, error, diffphase = phase_cross_correlation(reference_image,
                                                       shifted_image,
                                                       space="fourier")
    assert_allclose(result, -np.array(shift), atol=0.05)

    # subpixel precision now available for 3-D data

    subpixel_shift = (-2.3, 1.7, 5.4)
    shifted_image = fourier_shift(reference_image, subpixel_shift)
    result, error, diffphase = phase_cross_correlation(reference_image,
                                                       shifted_image,
                                                       upsample_factor=100,
                                                       space="fourier")
    assert_allclose(result, -np.array(subpixel_shift), atol=0.05)


def test_unknown_space_input():
    image = np.ones((5, 5))
    with pytest.raises(ValueError):
        phase_cross_correlation(
            image, image,
            space="frank")


def test_wrong_input():
    # Dimensionality mismatch
    image = np.ones((5, 5, 1))
    template = np.ones((5, 5))
    with pytest.raises(ValueError):
        phase_cross_correlation(template, image)

    # Size mismatch
    image = np.ones((5, 5))
    template = np.ones((4, 4))
    with pytest.raises(ValueError):
        phase_cross_correlation(template, image)

    # NaN values in data
    image = np.ones((5, 5))
    image[0][0] = np.nan
    template = np.ones((5, 5))
    with expected_warnings(
        [
            r"invalid value encountered in true_divide"
            + r"|"
            + r"invalid value encountered in divide"
            + r"|\A\Z"
        ]
    ):
        with pytest.raises(ValueError):
            phase_cross_correlation(template, image, return_error=True)


def test_4d_input_pixel():
    phantom = img_as_float(binary_blobs(length=32, n_dim=4))
    reference_image = fft.fftn(phantom)
    shift = (-2., 1., 5., -3)
    shifted_image = fourier_shift(reference_image, shift)
    result, error, diffphase = phase_cross_correlation(reference_image,
                                                       shifted_image,
                                                       space="fourier")
    assert_allclose(result, -np.array(shift), atol=0.05)


def test_4d_input_subpixel():
    phantom = img_as_float(binary_blobs(length=32, n_dim=4))
    reference_image = fft.fftn(phantom)
    subpixel_shift = (-2.3, 1.7, 5.4, -3.2)
    shifted_image = fourier_shift(reference_image, subpixel_shift)
    result, error, diffphase = phase_cross_correlation(reference_image,
                                                       shifted_image,
                                                       upsample_factor=10,
                                                       space="fourier")
    assert_allclose(result, -np.array(subpixel_shift), atol=0.05)


@pytest.mark.parametrize("return_error", [True, False, "always"])
@pytest.mark.parametrize("reference_mask", [None, True])
def test_phase_cross_correlation_deprecation(return_error, reference_mask):
    # For now, assert that phase_cross_correlation raises a warning that
    # returning only shifts is deprecated. In skimage 0.22, this test should be
    # updated for the deprecation of the return_error parameter.
    should_warn = (
        return_error is False
        or (return_error != "always" and reference_mask is True)
    )

    reference_image = np.ones((10, 10))
    moving_image = np.ones_like(reference_image)
    if reference_mask is True:
        # moving_mask defaults to reference_mask, passing moving_mask only is
        # not supported, so we don't need to test it
        reference_mask = np.ones_like(reference_image)

    if should_warn:
        msg = (
            "In scikit-image 0.22, phase_cross_correlation will start "
            "returning a tuple or 3 items (shift, error, phasediff) always. "
            "To enable the new return behavior and silence this warning, use "
            "return_error='always'."
        )
        with pytest.warns(FutureWarning, match=re.escape(msg)):
            out = phase_cross_correlation(
                reference_image=reference_image,
                moving_image=moving_image,
                return_error=return_error,
                reference_mask=reference_mask,
            )
        assert not isinstance(out, tuple)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = phase_cross_correlation(
                reference_image=reference_image,
                moving_image=moving_image,
                return_error=return_error,
                reference_mask=reference_mask,
            )
        assert isinstance(out, tuple)
        assert len(out) == 3


def test_mismatch_upsampled_region_size():
    with pytest.raises(ValueError):
        _upsampled_dft(
            np.ones((4, 4)),
            upsampled_region_size=[3, 2, 1, 4])


def test_mismatch_offsets_size():
    with pytest.raises(ValueError):
        _upsampled_dft(np.ones((4, 4)), 3,
                       axis_offsets=[3, 2, 1, 4])


@pytest.mark.parametrize(
        ('shift0', 'shift1'),
        itertools.product((100, -100, 350, -350), (100, -100, 350, -350)),
        )
def test_disambiguate_2d(shift0, shift1):
    image = eagle()[500:, 900:]  # use a highly textured part of image
    shift = (shift0, shift1)
    origin0 = []
    for s in shift:
        if s > 0:
            origin0.append(0)
        else:
            origin0.append(-s)
    origin1 = np.array(origin0) + shift
    slice0 = tuple(slice(o, o+450) for o in origin0)
    slice1 = tuple(slice(o, o+450) for o in origin1)
    reference = image[slice0]
    moving = image[slice1]
    computed_shift, _, _ = phase_cross_correlation(
            reference, moving, disambiguate=True, return_error='always'
            )
    np.testing.assert_equal(shift, computed_shift)


def test_disambiguate_zero_shift():
    """When the shift is 0, disambiguation becomes degenerate.

    Some quadrants become size 0, which prevents computation of
    cross-correlation. This test ensures that nothing bad happens in that
    scenario.
    """
    image = camera()
    computed_shift, _, _ = phase_cross_correlation(
            image, image, disambiguate=True, return_error='always'
            )
    assert computed_shift == (0, 0)
