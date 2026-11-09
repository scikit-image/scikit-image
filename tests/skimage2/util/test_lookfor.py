import _skimage2 as ski2
import pytest


# avoid bad interaction between SimpleITK, pytest, and lookfor
pytest.importorskip('SimpleITK')


def test_lookfor_basic(capsys):
    assert ski2.lookfor is ski2.util.lookfor

    ski2.util.lookfor("regionprops")
    search_results = capsys.readouterr().out
    assert "_skimage2.measure.regionprops" in search_results
    assert "_skimage2.measure.regionprops_table" in search_results
