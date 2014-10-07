from warnings import catch_warnings, simplefilter

def test_import_filter():
    with catch_warnings():
        simplefilter('ignore')
        from skimage import filter as F

    assert('sobel' in dir(F))
