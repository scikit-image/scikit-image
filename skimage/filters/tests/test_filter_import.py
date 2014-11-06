from skimage._shared.utils import all_warnings, skimage_deprecation
from warnings import catch_warnings, simplefilter
from numpy.testing import assert_warns

def import_filter():
    from skimage import filter as F
    assert('sobel' in dir(F))

def test_filter_import():
    with all_warnings():
        assert_warns(skimage_deprecation, import_filter)
