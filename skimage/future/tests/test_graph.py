import pytest


def test_future_graph_import_error():
    error_msg = "The API in skimage.future.graph was moved into skimage.graph"
    with pytest.raises(ModuleNotFoundError, match=error_msg):
        import skimage.future.graph

    with pytest.raises(ModuleNotFoundError, match=error_msg):
        from skimage.future import graph

    with pytest.raises(ModuleNotFoundError, match=error_msg):
        from skimage.future.graph import rag
