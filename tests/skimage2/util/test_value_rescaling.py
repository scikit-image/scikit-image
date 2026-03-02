import numpy as np
import pytest

from skimage.util.dtype import img_as_float
from skimage._shared._dependency_checks import is_wasm
from skimage._shared.testing import assert_stacklevel
from skimage._shared.utils import _supported_float_type
from skimage._shared.dtype import numeric_dtype_min_max

from skimage2.util._value_rescaling import minmax_rescale, _prescale_value_range


class TestMinmaxRescale:
    # Supported dtypes
    int_dtypes = [
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
    ]
    float_dtypes = [
        np.float16,
        np.float32,
        np.float64,
        # np.complex64,
        # np.complex128,
        # np.complex256,
    ]
    all_dtypes = int_dtypes + float_dtypes

    @pytest.mark.xfail(
        is_wasm, strict=False, reason="On WASM, NumPy does not report overflow errors"
    )
    @pytest.mark.parametrize("dtype", float_dtypes)
    def test_float_max(self, dtype):
        dtype_min, dtype_max = numeric_dtype_min_max(dtype)
        image = np.array(
            [dtype_min, dtype_min / 2, 0, dtype_max / 2, dtype_max], dtype=dtype
        )
        expected_dtype = _supported_float_type(dtype, allow_complex=True)
        expected = np.array([0, 0.25, 0.5, 0.75, 1], dtype=expected_dtype)

        if dtype == np.float16:
            # Special case: float16 is always scaled up to float32,
            # thereby avoiding over- & underflow issues and the related warning
            result = minmax_rescale(image)
        else:
            regex = "Overflow while attempting to rescale"
            with pytest.warns(RuntimeWarning, match=regex) as record:
                result = minmax_rescale(image)
            assert_stacklevel(record)

        assert image is not result
        assert result.dtype == expected_dtype
        np.testing.assert_equal(result, expected)

    @pytest.mark.parametrize("dtype", int_dtypes)
    def test_int_max(self, dtype):
        dtype_min, dtype_max = numeric_dtype_min_max(dtype)
        image = np.array([dtype_min, dtype_max], dtype=dtype)
        expected_dtype = _supported_float_type(dtype, allow_complex=True)
        expected = np.array([0, 1], dtype=expected_dtype)
        result = minmax_rescale(image)
        assert image is not result
        np.testing.assert_almost_equal(result, expected)

    @pytest.mark.parametrize("dtype", all_dtypes)
    def test_uniform(self, dtype):
        image = np.array([10, 10], dtype=dtype)
        expected_dtype = _supported_float_type(dtype, allow_complex=True)
        expected = np.array([0, 0], dtype=expected_dtype)

        with pytest.warns(RuntimeWarning, match="`image` is uniform") as record:
            result = minmax_rescale(image)
        assert_stacklevel(record)
        assert image is not result
        assert result.dtype == expected_dtype
        np.testing.assert_equal(result, expected)

    @pytest.mark.parametrize("dtype", float_dtypes)
    def test_nan(self, dtype):
        image = np.array([np.nan, -1, 0, 1], dtype=dtype)
        with pytest.raises(ValueError, match="`image` contains NaN"):
            minmax_rescale(image)

    @pytest.mark.parametrize("dtype", float_dtypes)
    @pytest.mark.filterwarnings(
        "ignore:Dividing by 2 before scaling to avoid over-/underflow:RuntimeWarning"
    )
    def test_inf(self, dtype):
        image = np.array([-np.inf, -1, 0, 1, np.inf], dtype=dtype)
        with pytest.raises(ValueError, match="`image` contains inf"):
            minmax_rescale(image)


class TestPrescaleValueRange:
    @pytest.mark.parametrize("dtype", TestMinmaxRescale.all_dtypes)
    def test_mode_none(self, dtype):
        dtype_min, dtype_max = numeric_dtype_min_max(dtype)
        image = np.array([dtype_min, 0, dtype_max], dtype=dtype)

        result = _prescale_value_range(image, mode="none")
        assert result is not image
        assert result.dtype == dtype
        np.testing.assert_equal(result, image)

    @pytest.mark.parametrize("dtype", TestMinmaxRescale.all_dtypes)
    def test_mode_legacy(self, dtype):
        dtype_min, dtype_max = numeric_dtype_min_max(dtype)
        image = np.array([dtype_min, 0, dtype_max], dtype=dtype)
        expected = img_as_float(image)

        result = _prescale_value_range(image, mode="legacy")
        np.testing.assert_equal(result, expected)
        assert result.dtype == expected.dtype

    @pytest.mark.filterwarnings("ignore:Overflow while attempting to rescale")
    @pytest.mark.parametrize("dtype", TestMinmaxRescale.all_dtypes)
    def test_mode_minmax(self, dtype):
        dtype_min, dtype_max = numeric_dtype_min_max(dtype)
        image = np.array([dtype_min, 0, dtype_max], dtype=dtype)
        expected = minmax_rescale(image)

        result = _prescale_value_range(image, mode="minmax")
        np.testing.assert_equal(result, expected)
        assert result.dtype == expected.dtype

    @pytest.mark.parametrize("mode", ["dtype", False, ""])
    def test_mode_unsupported(self, mode):
        image = np.array([-100, 0, 200], dtype=float)
        with pytest.raises(ValueError, match="unsupported mode"):
            _prescale_value_range(image, mode=mode)
