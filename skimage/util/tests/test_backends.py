import pytest
from skimage.util import _backends


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


@pytest.fixture
def no_backends(monkeypatch):
    """Mock backend setup with no backends"""

    def mock_no_backends():
        return {}

    monkeypatch.setattr(_backends, "all_backends", mock_no_backends)


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
