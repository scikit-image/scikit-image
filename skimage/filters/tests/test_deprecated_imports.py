from warnings import catch_warnings, simplefilter
from ..._shared._warnings import expected_warnings
from ...data import moon


def test_filter_import():
    with catch_warnings():
        simplefilter('ignore')
        from skimage import filter as F

    assert('sobel' in dir(F))
    assert F._import_warned


def test_canny_import():
    data = moon()
    with expected_warnings(['skimage.feature.canny']):
        from skimage.filters import canny
        canny(data)
