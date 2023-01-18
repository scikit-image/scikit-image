import pytest


def test_future_graph_import_error():
    error_msg = (
        "The `skimage.future.graph` submodule was moved to `skimage.graph` in "
        "v0.20. Please update your import paths accordingly."
    )
    with pytest.raises(ModuleNotFoundError, match=error_msg):
        import skimage.future.graph

    with pytest.raises(ModuleNotFoundError, match=error_msg):
        from skimage.future import graph

    with pytest.raises(ModuleNotFoundError, match=error_msg):
        from skimage.future.graph import rag
