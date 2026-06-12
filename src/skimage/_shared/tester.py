from _skimage2._shared.tester import PytestTester as PytestTester  # noqa: F401

__all__ = ['PytestTester']

from skimage._docutils import bind_namespace

bind_namespace(globals())
