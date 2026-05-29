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
                    "Backend only implements the 'foo' function."
                    f" Called with '{name}'"
                )

            return fake_foo

        def can_has(self, name, *args, **kwargs):
            if not name.endswith(":foo"):
                raise ValueError(
                    "Backend only implements the 'foo' function."
                    f" Called with '{name}'"
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

    def mock_all_backends():
        return {
            "fake2": {
                "implementation": BackendEntryPoint2(),
                "info": _backends.BackendInformation([f"{__name__}:foo"]),
            },
            "fake1": {
                "implementation": BackendEntryPoint1(),
                "info": _backends.BackendInformation([f"{__name__}:foo"]),
            },
        }

    monkeypatch.setattr(_backends, "all_backends", mock_all_backends)
    monkeypatch.setattr(_backends, "public_api_module", mock_public_api_module)


@pytest.fixture
def no_backends(monkeypatch):
    """Mock backend setup with no backends"""

    def mock_no_backends():
        return {}

    monkeypatch.setattr(_backends, "all_backends", mock_no_backends)
    monkeypatch.setattr(_backends, "public_api_module", mock_public_api_module)


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
        # Checking for fake1 means we also check that the backends are
        # used in the correct priority/order
        match="Call to.*:foo' was dispatched to the 'fake1' backend",
    ):
        r = foo(42)

    assert r == 42 * 3


# Tell tests below that we mean a skimage implementation, not the
# implementation imported from skimage2.
_SKI_ENTROPY = _backends.dispatchable_shim(
    skimage.filters.rank.entropy, 'skimage.filters.rank'
)


@pytest.mark.parametrize(
    "func, expected, exp_func",
    [
        (skimage.metrics.mean_squared_error, "skimage.metrics", None),
        # Imported from ski2
        (skimage.io.concatenate_images, "_skimage2.io", None),
        # Imported from skimage, as above, but finds ski2 implementation.
        (_SKI_ENTROPY, "skimage.filters.rank", skimage.filters.rank.entropy),
    ],
)
def test_module_name_determination(func, expected, exp_func):
    exp_func = func if exp_func is None else exp_func
    module_name = _backends.public_api_module(func)

    assert module_name == expected

    mod = importlib.import_module(module_name)
    assert getattr(mod, func.__name__) is exp_func


def test_backend_information_rejects_skimage2_keys():
    with pytest.raises(ValueError, match="_skimage2"):
        _backends.BackendInformation(["_skimage2.metrics:mean_squared_error"])
