from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from skimage import data, filters
from skimage._shared.utils import _supported_float_type


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64,
                                   np.uint8])
def test_homomorphic(dtype):
    eagle_lowres = data.eagle()[::4, ::4]
    eagle_lowres = eagle_lowres.astype(dtype)
    func_kwargs = dict(cutoff_frequency_ratio=0.02, npad=32,
                       amplitude_range=(0.3, 1))
    eagle_filtered = filters.homomorphic(filters.butterworth, eagle_lowres,
                                         func_kwargs=func_kwargs)
    expected_dtype = _supported_float_type(eagle_lowres.dtype)
    assert eagle_filtered.dtype == expected_dtype
    # output does not have range exceeding the input
    assert eagle_filtered.max() <= eagle_lowres.max()
    # filtered image will have lower norm (energy)
    assert np.linalg.norm(eagle_filtered) < np.linalg.norm(eagle_lowres)

    # verify that passing func_kwargs above is equivalent to the following
    filter_func = partial(filters.butterworth, **func_kwargs)
    eagle_filtered2 = filters.homomorphic(filter_func, eagle_lowres)
    assert_array_equal(eagle_filtered, eagle_filtered2)
