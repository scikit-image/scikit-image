from warnings import catch_warnings, simplefilter


def test_filter_import():
    with catch_warnings():
        simplefilter('ignore')
        from skimage import filter as F

    assert('sobel' in dir(F))
    assert F._import_warned


def test_canny_import():
    with catch_warnings():
        simplefilter('ignore')
        from skimage.filters import canny

    assert('canny' in dir(F))
    assert F._import_warned
