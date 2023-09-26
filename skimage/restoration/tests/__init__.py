import pytest

try:
    import pywt
except ImportError:
    xfail_without_pywt = pytest.mark.xfail(
        reason="optional dependency PyWavelets is not installed",
        raises=ImportError,
    )
else:
    def skip_without_pywt(func):
        return func
