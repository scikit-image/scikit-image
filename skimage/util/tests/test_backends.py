import importlib

import pytest

import skimage.metrics
from skimage.util import _backends


def mock_get_module_name(func):
    # Unlike the real `get_module_name` this returns the actual
    # full module name and does not perform a sanity check
    # because that test would fail for the functions we define
    # inside our tests.
    return func.__module__


@pytest.fixture
def fake_backends(monkeypatch):
    """Mock backend setup

    Two backends, with one function each.
    """

    class Backend1:
        def get_implementation(self, name):
            def fake_foo(x):
                return x * 3

            if not name.endswith(".foo"):
                raise ValueError(
                    "Backend only implements the 'foo' function."
                    f" Called with '{name}'"
                )

            return fake_foo

        def can_has(self, name, *args, **kwargs):
            if not name.endswith(".foo"):
                raise ValueError(
                    "Backend only implements the 'foo' function."
                    f" Called with '{name}'"
                )
            return True

    class Backend2:
        def get_implementation(self, name):
            def fake_foo(x):
                return x * 4

            if not name.endswith(".foo"):
                return None

            return fake_foo

        def can_has(self, name, *args, **kwargs):
            if name.endswith(".foo"):
                return True
            else:
                return False

    class BackendEntryPoint1:
        def load(self):
            return Backend1()

    class BackendEntryPoint2:
        def load(self):
            return Backend2()

    def mock_all_backends():
        return {
            "fake1": {"implementation": BackendEntryPoint1()},
            "fake2": {"implementation": BackendEntryPoint2()},
        }

    monkeypatch.setattr(_backends, "all_backends", mock_all_backends)
    monkeypatch.setattr(_backends, "get_module_name", mock_get_module_name)


@pytest.fixture
def no_backends(monkeypatch):
    """Mock backend setup with no backends"""

    def mock_no_backends():
        return {}

    monkeypatch.setattr(_backends, "all_backends", mock_no_backends)
    monkeypatch.setattr(_backends, "get_module_name", mock_get_module_name)


def test_no_notification_without_backends(no_backends):
    # Check that no DispatchNotification is raised when no backend
    # is installed.
    @_backends.dispatchable
    def foo(x):
        return x * 2

    r = foo(42)

    assert r == 42 * 2


def test_no_dispatching_when_disabled(fake_backends, monkeypatch):
    monkeypatch.setenv("SKIMAGE_NO_DISPATCHING", "1")

    @_backends.dispatchable
    def foo(x):
        return x * 2

    r = foo(42)

    assert r == 42 * 2


def test_notification_raised(fake_backends):
    @_backends.dispatchable
    def foo(x):
        return x * 2

    with pytest.warns(
        _backends.DispatchNotification,
        match="Call to.*foo' was dispatched to the 'fake1' backend",
    ):
        r = foo(42)

    assert r == 42 * 3


@pytest.mark.parametrize(
    "func, expected",
    [
        (skimage.metrics.mean_squared_error, "skimage.metrics"),
        (skimage.io.concatenate_images, "skimage.io"),
    ],
)
def test_module_name_determination(func, expected):
    module_name = _backends.get_module_name(func)

    assert module_name == expected

    mod = importlib.import_module(module_name)
    assert getattr(mod, func.__name__) is func
