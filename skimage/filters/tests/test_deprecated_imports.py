from ..._shared._warnings import expected_warnings
from ...data import moon


def test_canny_import():
    data = moon()
    with expected_warnings(['skimage.feature.canny']):
        from skimage.filters import canny
        canny(data)
