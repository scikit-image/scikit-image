import os

from scipy._lib._array_api import (
    xp_assert_close,
    xp_assert_equal,
    assert_array_almost_equal,
    assert_almost_equal,

    array_namespace,
    xp_swapaxes,

    default_xp,
    _xp_copy_to_numpy,

    is_torch,
    is_numpy,
)


# To enable array API and strict array-like input validation
SCIPY_ARRAY_API: str | bool = os.environ.get("SCIPY_ARRAY_API", False)
# To control the default device - for use in the test suite only
SCIPY_DEVICE = os.environ.get("SCIPY_DEVICE", "cpu")
