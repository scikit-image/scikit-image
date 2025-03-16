import importlib

import pytest

import skimage.metrics
from skimage.util import _backends


def mock_public_api_module(func):
    # Unlike the real `public_api_module` this returns the actual
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

            if not name.endswith(":foo"):
                raise ValueError(
                    f"Backend only implements the 'foo' function. Called with '{name}'"
                )

            return fake_foo

        def can_has(self, name, *args, **kwargs):
            if not name.endswith(":foo"):
                raise ValueError(
                    f"Backend only implements the 'foo' function. Called with '{name}'"
                )
            return True

    class Backend2:
        def get_implementation(self, name):
            def fake_foo(x):
                return x * 4

            if not name.endswith(":foo"):
                return None

            return fake_foo

        def can_has(self, name, *args, **kwargs):
            if name.endswith(":foo"):
                return True
            else:
                return False

    class BackendEntryPoint1:
        def load(self):
            return Backend1()

    class BackendEntryPoint2:
        def load(self):
            return Backend2()

    def mock_all_backends_with_eps_combined():
        return {
            "fake2": {
                "skimage_backends_ep_obj": BackendEntryPoint2(),
                "info": _backends.BackendInformation(
                    ["skimage.util.tests.test_backends:foo"]
                ),
            },
            "fake1": {
                "skimage_backends_ep_obj": BackendEntryPoint1(),
                "info": _backends.BackendInformation(
                    ["skimage.util.tests.test_backends:foo"]
                ),
            },
        }

    monkeypatch.setattr(
        _backends, "all_backends_with_eps_combined", mock_all_backends_with_eps_combined
    )
    monkeypatch.setattr(_backends, "public_api_module", mock_public_api_module)
    monkeypatch.setenv("SKIMAGE_DISPATCHING", "True")
    monkeypatch.setenv("SKIMAGE_BACKEND_PRIORITY", "fake1, fake2")


@pytest.fixture
def no_backends(monkeypatch):
    """Mock backend setup with no backends"""

    def mock_no_backends():
        return {}

    monkeypatch.setattr(_backends, "all_backends_with_eps_combined", mock_no_backends)
    monkeypatch.setattr(_backends, "public_api_module", mock_public_api_module)
    monkeypatch.setenv("SKIMAGE_DISPATCHING", "True")


def test_notification_without_backends(monkeypatch, no_backends):
    # Check that no DispatchNotification is raised when no backend
    # is installed.
    @_backends.dispatchable
    def foo(x):
        return x * 2

    r = foo(42)

    assert r == 42 * 2

    # no backends installed but `SKIMAGE_DISPATCHING` is "True"
    monkeypatch.setenv("SKIMAGE_BACKEND_PRIORITY", "fake1")
    with pytest.warns(
        _backends.DispatchNotification,
        match="Call to.*:foo' was not dispatched.",
    ):
        r = foo(42)

    assert r == 42 * 2


def test_when_dispatching_disabled(fake_backends, monkeypatch):
    monkeypatch.setenv("SKIMAGE_DISPATCHING", "False")

    @_backends.dispatchable
    def foo(x):
        return x * 2

    r = foo(42)

    assert r == 42 * 2


def test_notification_with_backends(monkeypatch, fake_backends):
    @_backends.dispatchable
    def foo(x):
        return x * 2

    with pytest.warns(
        _backends.DispatchNotification,
        # Checking for fake1 means we also check that the backends are
        # used in the correct priority/order
        match="Call to.*:foo' was dispatched to the 'fake1' backend",
    ):
        r = foo(42)

    assert r == 42 * 3

    monkeypatch.setenv("SKIMAGE_BACKEND_PRIORITY", "fake3")  # "fake3" is not a backend.
    with pytest.warns(
        _backends.DispatchNotification,
        match="Call to.*:foo' was not dispatched.",
    ):
        r = foo(42)

    assert r == 42 * 2


@pytest.mark.parametrize(
    "func, expected",
    [
        (skimage.metrics.mean_squared_error, "skimage.metrics"),
        (skimage.io.concatenate_images, "skimage.io"),
        (skimage.filters.rank.entropy, "skimage.filters.rank"),
    ],
)
def test_module_name_determination(func, expected):
    module_name = _backends.public_api_module(func)

    assert module_name == expected

    mod = importlib.import_module(module_name)
    assert getattr(mod, func.__name__) is func


@pytest.mark.parametrize(
    "env_value, output", [("True", True), ("False", False), ("xyz", False)]
)
def test_get_skimage_dispatching(monkeypatch, env_value, output):
    """Test the behavior of get_skimage_dispatching with different environment variable values."""
    monkeypatch.setenv("SKIMAGE_DISPATCHING", env_value)
    assert _backends.get_skimage_dispatching() == output


@pytest.mark.parametrize(
    "env_value, output",
    [
        ("False", False),
        (
            "backend1",
            [
                "backend1",
            ],
        ),
        ("backend1,backend2, backend3", ["backend1", "backend2", "backend3"]),
    ],
)
def test_get_skimage_backend_priority(monkeypatch, env_value, output):
    """Test the behavior of get_skimage_backends with different environment variable values."""
    monkeypatch.setenv("SKIMAGE_BACKEND_PRIORITY", env_value)
    assert _backends.get_skimage_backend_priority() == output
