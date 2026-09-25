import skimage as ski
import pytest


# avoid bad interaction between SimpleITK, pytest, and lookfor
pytest.importorskip('SimpleITK')


def test_lookfor_basic(capsys):
    assert ski.lookfor is ski.util.lookfor

    ski.util.lookfor("regionprops")
    search_results = capsys.readouterr().out
    assert "skimage.measure.regionprops" in search_results
    assert "skimage.measure.regionprops_table" in search_results
