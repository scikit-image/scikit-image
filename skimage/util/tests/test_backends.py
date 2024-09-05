import pytest
from skimage.util import _backends


@pytest.fixture
def mock_fake_backends(monkeypatch):
    """Mock backend setup

    One backend, with one function.
    """

    class Backend:
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

    class BackendEntryPoint:
        def load(self):
            return Backend()

    def mock_all_backends():
        return {"fake": {"implementation": BackendEntryPoint()}}

    monkeypatch.setattr(_backends, "all_backends", mock_all_backends)


@pytest.fixture
def mock_no_backends(monkeypatch):
    """Mock backend setup with no backends"""

    def mock_no_backends():
        return {}

    monkeypatch.setattr(_backends, "all_backends", mock_no_backends)


def test_no_notification_without_backends(mock_no_backends):
    # Check that no DispatchNotification is raised when no backend
    # is installed.
    @_backends.dispatchable
    def foo(x):
        return x * 2

    r = foo(42)

    assert r == 42 * 2


def test_no_dispatching_when_disabled(mock_fake_backends, monkeypatch):
    monkeypatch.setenv("SKIMAGE_NO_DISPATCHING", "1")

    @_backends.dispatchable
    def foo(x):
        return x * 2

    r = foo(42)

    assert r == 42 * 2


def test_notification_raised(mock_fake_backends):
    @_backends.dispatchable
    def foo(x):
        return x * 2

    with pytest.warns(_backends.DispatchNotification):
        r = foo(42)

    assert r == 42 * 3


def test_all_backends():
    _backends.all_backends()

    _backends.all_backends.cache_clear()
