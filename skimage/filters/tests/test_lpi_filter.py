import numpy as np
import pytest
from numpy.testing import assert_, assert_equal

from skimage._shared.utils import _supported_float_type
from skimage.data import camera
from skimage.filters.lpi_filter import LPIFilter2D, filter_inverse, wiener


class TestLPIFilter2D:

    img = camera()[:50, :50]

    def filt_func(self, r, c):
        return np.exp(-np.hypot(r, c) / 1)

    def setup_method(self):
        self.f = LPIFilter2D(self.filt_func)

    @pytest.mark.parametrize(
        'c_slice', [slice(None), slice(0, -5), slice(0, -20)]
    )
    def test_ip_shape(self, c_slice):
        x = self.img[:, c_slice]
        assert_equal(self.f(x).shape, x.shape)

    @pytest.mark.parametrize(
        'dtype', [np.uint8, np.float16, np.float32, np.float64]
    )
    def test_filter_inverse(self, dtype):
        img = self.img.astype(dtype, copy=False)
        expected_dtype = _supported_float_type(dtype)

        F = self.f(img)
        assert F.dtype == expected_dtype

        g = filter_inverse(F, predefined_filter=self.f)
        assert g.dtype == expected_dtype
        assert_equal(g.shape, self.img.shape)

        g1 = filter_inverse(F[::-1, ::-1], predefined_filter=self.f)
        assert_((g - g1[::-1, ::-1]).sum() < 55)

        # test cache
        g1 = filter_inverse(F[::-1, ::-1], predefined_filter=self.f)
        assert_((g - g1[::-1, ::-1]).sum() < 55)

        g1 = filter_inverse(F[::-1, ::-1], self.filt_func)
        assert_((g - g1[::-1, ::-1]).sum() < 55)

    @pytest.mark.parametrize(
        'dtype', [np.uint8, np.float16, np.float32, np.float64]
    )
    def test_wiener(self, dtype):

        img = self.img.astype(dtype, copy=False)
        expected_dtype = _supported_float_type(dtype)

        F = self.f(img)
        assert F.dtype == expected_dtype

        g = wiener(F, predefined_filter=self.f)
        assert g.dtype == expected_dtype
        assert_equal(g.shape, self.img.shape)

        g1 = wiener(F[::-1, ::-1], predefined_filter=self.f)
        assert_((g - g1[::-1, ::-1]).sum() < 1)

        g1 = wiener(F[::-1, ::-1], self.filt_func)
        assert_((g - g1[::-1, ::-1]).sum() < 1)

    def test_non_callable(self):
        with pytest.raises(ValueError):
            LPIFilter2D(None)
