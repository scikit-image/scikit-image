from warnings import catch_warnings, simplefilter


def test_filter_import():
    with catch_warnings():
        simplefilter('always')
        from skimage import filter as F

    assert('sobel' in dir(F))
    assert any(['has been renamed' in w
                for (w, _, _) in F.__warningregistry__])
