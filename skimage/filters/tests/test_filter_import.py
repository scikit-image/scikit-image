from warnings import catch_warnings, simplefilter

from skimage.filters import median

def test_filter_import():
    with catch_warnings():
        simplefilter('ignore')
        from skimage import filter as F

    assert('sobel' in dir(F))
    assert F._import_warned
