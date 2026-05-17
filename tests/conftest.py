# Fixtures for doctests
from pathlib import Path
import pytest
from time import perf_counter
from datetime import timedelta
import json
import os
import sys
import sysconfig
from typing import Literal
from pytest_pretty import CustomTerminalReporter

from _pytest.terminal import TerminalReporter
from _pytest.pathlib import bestrelpath

import numpy as np
from _skimage2.util._array_api import (
    SCIPY_ARRAY_API, SCIPY_DEVICE, array_namespace, default_xp
)


FREE_THREADED_BUILD = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
GIL_ENABLED_AT_START = getattr(sys, "_is_gil_enabled", lambda: True)()


class SKTerminalReporter(CustomTerminalReporter):
    """Custom terminal reporter to display test runtimes.

    It appends the cumulative runtime each time a new file is tested.
    """

    currentfspath: Path | None
    _start_time: float | None

    def write_fspath_result(self, nodeid: str, res, **markup: bool) -> None:
        if getattr(self, '_start_time', None) is None:
            self._start_time = perf_counter()

        fspath = self.config.rootpath / nodeid.split('::')[0]
        if fspath != self.currentfspath:
            if self.currentfspath is not None and self._show_progress_info:
                # call method to write information about progress
                # padding spaces and percentage information
                self._write_progress_information_filling_space()
                if os.environ.get('CI', False):
                    # write time elapsed since the beginning of the test suite
                    elapsed = timedelta(seconds=int(perf_counter() - self._start_time))
                    self.write(f' [{elapsed}]')

            self.currentfspath = fspath
            relfspath = bestrelpath(self.startpath, fspath)
            self._tw.line()
            self.write(relfspath + ' ')

        self.write(res, flush=True, **markup)

    def short_test_summary(self):
        # Don't use table-based summary from pytest_pretty,
        # use pytest's original one that won't truncate long file and test names
        TerminalReporter.short_test_summary(self)


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    # Get the standard terminal reporter plugin and replace it with ours
    standard_reporter = config.pluginmanager.getplugin('terminalreporter')
    custom_reporter = SKTerminalReporter(config, sys.stdout)
    if standard_reporter._session is not None:
        # if session is already set we need to copy them from
        # the previous reporter
        custom_reporter._session = standard_reporter._session
    config.pluginmanager.unregister(standard_reporter)
    config.pluginmanager.register(custom_reporter, 'terminalreporter')

    config.addinivalue_line("markers",
        "array_api_backends: test iterates on all array API backends")
    config.addinivalue_line("markers",
        ("skip_xp_backends(backends, reason=None, np_only=False, cpu_only=False, " +
         "eager_only=False, exceptions=None): mark the desired skip configuration " +
         "for the `skip_xp_backends` fixture"))
    config.addinivalue_line("markers",
        ("xfail_xp_backends(backends, reason=None, np_only=False, cpu_only=False, " +
         "eager_only=False, exceptions=None): mark the desired xfail configuration " +
         "for the `xfail_xp_backends` fixture"))

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if FREE_THREADED_BUILD and not GIL_ENABLED_AT_START and sys._is_gil_enabled():
        tr = terminalreporter
        tr.ensure_newline()
        tr.section("GIL re-enabled", sep="=", red=True, bold=True)
        tr.line("The GIL was re-enabled at runtime during the tests.")
        tr.line("This can happen with no test failures if the RuntimeWarning")
        tr.line("raised by Python when this happens is filtered by a test.")
        tr.line("")
        tr.line("Please ensure all new C and C++ extensions declare support")
        tr.line("for running without the GIL.")
        pytest.exit("GIL re-enabled during tests", returncode=1)


# Array API backend handling
# XXX parroted from SciPy; deduplicate?
xp_known_backends = {'numpy', 'array_api_strict', 'torch', 'cupy', 'jax.numpy',
                     'dask.array'}
xp_available_backends = [
    pytest.param(np, id='numpy', marks=pytest.mark.array_api_backends)
]
xp_skip_cpu_only_backends = set()
xp_skip_eager_only_backends = set()

if SCIPY_ARRAY_API:
    # fill the dict of backends with available libraries
    try:
        import array_api_strict
        xp_available_backends.append(
            pytest.param(array_api_strict, id='array_api_strict',
                         marks=pytest.mark.array_api_backends))
        if array_api_strict.__version__ < '2.3':   # XXX: packaging.version
            raise ImportError("array-api-strict must be >= version 2.3")
        array_api_strict.set_array_api_strict_flags(
            api_version='2025.12'
        )
    except ImportError:
        pass

    try:
        import torch  # type: ignore[import-not-found]
        xp_available_backends.append(
            pytest.param(torch, id='torch',
            marks=pytest.mark.array_api_backends))
        torch.set_default_device(SCIPY_DEVICE)
        if SCIPY_DEVICE != "cpu":
            xp_skip_cpu_only_backends.add('torch')

        # default to float64 unless explicitly requested
        default = os.getenv('SCIPY_DEFAULT_DTYPE', default='float64')
        if default == 'float64':
            torch.set_default_dtype(torch.float64)
        elif default != "float32":
            raise ValueError(
                "SCIPY_DEFAULT_DTYPE env var, if set, can only be either 'float64' "
               f"or 'float32'. Got '{default}' instead."
            )
    except ImportError:
        pass

    try:
        import cupy  # type: ignore[import-not-found]
        # Note: cupy disregards SCIPY_DEVICE and always runs on cuda.
        # It will fail to import if you don't have CUDA hardware and drivers.
        xp_available_backends.append(
            pytest.param(cupy, id='cupy',
            marks=pytest.mark.array_api_backends))
        xp_skip_cpu_only_backends.add('cupy')

        # this is annoying in CuPy 13.x
        warnings.filterwarnings(
            'ignore', 'cupyx.jit.rawkernel is experimental', category=FutureWarning
        )
        from cupyx.scipy import signal
        del signal
    except ImportError:
        pass

    try:
        import jax.numpy  # type: ignore[import-not-found]
        
        xp_available_backends.append(
            pytest.param(jax.numpy, id='jax.numpy',
            marks=[pytest.mark.array_api_backends,
                   # Uses xpx.testing.patch_lazy_xp_functions to monkey-patch module
                   pytest.mark.thread_unsafe]))

        jax.config.update("jax_enable_x64", True)
        # Make sure JAX won't default to less accurate TensorFloat32 precision
        # in matmuls with float32 inputs on GPUs that support this floating
        # point format.
        jax.config.update("jax_default_matmul_precision", "float32")
        jax.config.update("jax_default_device", jax.devices(SCIPY_DEVICE)[0])
        if SCIPY_DEVICE != "cpu":
            xp_skip_cpu_only_backends.add('jax.numpy')
        # JAX can be eager or lazy (when wrapped in jax.jit). However it is
        # recommended by upstream devs to assume it's always lazy.
        xp_skip_eager_only_backends.add('jax.numpy')
    except ImportError:
        pass


    xp_available_backend_ids = {p.id for p in xp_available_backends}
    assert not xp_available_backend_ids - xp_known_backends

    # by default, use all available backends
    if (
        isinstance(SCIPY_ARRAY_API, str)
        and SCIPY_ARRAY_API.lower() not in ("1", "true", "all")
    ):
        SCIPY_ARRAY_API_ = set(json.loads(SCIPY_ARRAY_API))
        if SCIPY_ARRAY_API_ != {'all'}:
            if SCIPY_ARRAY_API_ - xp_available_backend_ids:
                msg = ("'--array-api-backend' must be in "
                       f"{xp_available_backend_ids}; got {SCIPY_ARRAY_API_}")
                raise ValueError(msg)
            # Only select a subset of backends
            xp_available_backends = [
                param for param in xp_available_backends
                if param.id in SCIPY_ARRAY_API_
            ]

@pytest.fixture(params=xp_available_backends)
def xp(request):
    """Run the test that uses this fixture on each available array API library.

    You can select all and only the tests that use the `xp` fixture by
    passing `-m array_api_backends` to pytest.

    You can select where individual tests run through the `@skip_xp_backends`,
    `@xfail_xp_backends`, and `@skip_xp_invalid_arg` pytest markers.

    Please read: https://docs.scipy.org/doc/scipy/dev/api-dev/array_api.html#adding-tests
    """
    # Read all @pytest.marks.skip_xp_backends markers that decorate to the test,
    # if any, and raise pytest.skip() if the current xp is in the list.
    skip_or_xfail_xp_backends(request, "skip")
    # Read all @pytest.marks.xfail_xp_backends markers that decorate the test,
    # if any, and raise pytest.xfail() if the current xp is in the list.
    skip_or_xfail_xp_backends(request, "xfail")

    # Check if ``uses_xp_capabilities`` mark is present.
    # ``scipy._lib._array_api.make_xp_pytest_marks``, which draws from
    # ``xp_capabilities``, will set ``pytest.mark.uses_xp_capabilities(True)``.
    # Tests which are unconverted or which are for private functions without
    # ``xp_capabilities`` entries should have
    # ``pytest.mark.uses_xp_capabilities(False)`` explicitly set.
##    if request.node.get_closest_marker("uses_xp_capabilities") is None:
##        warnings.warn(
##            "test uses `xp` fixture without drawing from `xp_capabilities` "
##            " but is not explicitly marked with"
##            " ``pytest.mark.uses_xp_capabilities(False)``",
##            stacklevel=0,
##        )

    xp = request.param
    # Potentially wrap namespace with array_api_compat
    xp = array_namespace(xp.empty(0))

    if SCIPY_ARRAY_API:
        # If xp==jax.numpy, wrap tested functions in jax.jit
        # If xp==dask.array, wrap tested functions to test that graph is not computed
##        with patch_lazy_xp_functions(request=request, xp=request.param):
            # Throughout all calls to assert_almost_equal, assert_array_almost_equal,
            # and xp_assert_* functions, test that the array namespace is xp in both
            # the expected and actual arrays. This is to detect the case where both
            # arrays are erroneously just plain numpy while xp is something else.
            with default_xp(xp):
                yield xp
    else:
        yield xp


skip_xp_invalid_arg = pytest.mark.skipif(SCIPY_ARRAY_API,
    reason = ('Test involves masked arrays, object arrays, or other types '
              'that are not valid input when `SCIPY_ARRAY_API` is used.'))


def _backends_kwargs_from_request(request, skip_or_xfail):
    """A helper for {skip,xfail}_xp_backends.

    Return dict of {backend to skip/xfail: top reason to skip/xfail it}
    """
    markers = list(request.node.iter_markers(f'{skip_or_xfail}_xp_backends'))
    reasons = {backend: [] for backend in xp_known_backends}

    for marker in markers:
        invalid_kwargs = set(marker.kwargs) - {
            "cpu_only", "np_only", "eager_only", "reason", "exceptions"}
        if invalid_kwargs:
            raise TypeError(f"Invalid kwargs: {invalid_kwargs}")

        exceptions = set(marker.kwargs.get('exceptions', []))
        invalid_exceptions = exceptions - xp_known_backends
        if (invalid_exceptions := list(exceptions - xp_known_backends)):
            raise ValueError(f"Unknown backend(s): {invalid_exceptions}; "
                             f"must be a subset of {list(xp_known_backends)}")

        if marker.kwargs.get('np_only', False):
            reason = marker.kwargs.get("reason") or "do not run with non-NumPy backends"
            for backend, backend_reasons in reasons.items():
                if backend != 'numpy' and backend not in exceptions:
                    backend_reasons.append(reason)

        elif marker.kwargs.get('cpu_only', False):
            reason = marker.kwargs.get("reason") or (
                "no array-agnostic implementation or delegation available "
                "for this backend and device")
            for backend in xp_skip_cpu_only_backends - exceptions:
                reasons[backend].append(reason)

        elif marker.kwargs.get('eager_only', False):
            reason = marker.kwargs.get("reason") or (
                "eager checks not executed on lazy backends")
            for backend in xp_skip_eager_only_backends - exceptions:
                reasons[backend].append(reason)

        # add backends, if any
        if len(marker.args) == 1:
            backend = marker.args[0]
            if backend not in xp_known_backends:
                raise ValueError(f"Unknown backend: {backend}; "
                                 f"must be one of {list(xp_known_backends)}")
            reason = marker.kwargs.get("reason") or (
                f"do not run with array API backend: {backend}")
            # reason overrides the ones from cpu_only, np_only, and eager_only.
            # This is regardless of order of appearence of the markers.
            reasons[backend].insert(0, reason)

            for kwarg in ("cpu_only", "np_only", "eager_only", "exceptions"):
                if kwarg in marker.kwargs:
                    raise ValueError(f"{kwarg} is mutually exclusive with {backend}")

        elif len(marker.args) > 1:
            raise ValueError(
                f"Please specify only one backend per marker: {marker.args}"
            )

    return {backend: backend_reasons[0]
            for backend, backend_reasons in reasons.items()
            if backend_reasons}


def skip_or_xfail_xp_backends(request: pytest.FixtureRequest,
                              skip_or_xfail: Literal['skip', 'xfail']) -> None:
    """
    Helper of the `xp` fixture.
    Skip or xfail based on the ``skip_xp_backends`` or ``xfail_xp_backends`` markers.

    See the "Support for the array API standard" docs page for usage examples.

    Usage
    -----
    ::
        skip_xp_backends = pytest.mark.skip_xp_backends
        xfail_xp_backends = pytest.mark.xfail_xp_backends
        ...

        @skip_xp_backends(backend, *, reason=None)
        @skip_xp_backends(*, cpu_only=True, exceptions=(), reason=None)
        @skip_xp_backends(*, eager_only=True, exceptions=(), reason=None)
        @skip_xp_backends(*, np_only=True, exceptions=(), reason=None)

        @xfail_xp_backends(backend, *, reason=None)
        @xfail_xp_backends(*, cpu_only=True, exceptions=(), reason=None)
        @xfail_xp_backends(*, eager_only=True, exceptions=(), reason=None)
        @xfail_xp_backends(*, np_only=True, exceptions=(), reason=None)

    Parameters
    ----------
    backend : str, optional
        Backend to skip/xfail, e.g. ``"torch"``.
        Mutually exclusive with ``cpu_only``, ``eager_only``, and ``np_only``.
    cpu_only : bool, optional
        When ``True``, the test is skipped/xfailed on non-CPU devices,
        minus exceptions. Mutually exclusive with ``backend``.
    eager_only : bool, optional
        When ``True``, the test is skipped/xfailed for lazy backends, e.g. those
        with major caveats when invoking ``__array__``, ``__bool__``, ``__float__``,
        or ``__complex__``, minus exceptions. Mutually exclusive with ``backend``.
    np_only : bool, optional
        When ``True``, the test is skipped/xfailed for all backends other
        than the default NumPy backend and the exceptions.
        Mutually exclusive with ``backend``. Implies ``cpu_only`` and ``eager_only``.
    reason : str, optional
        A reason for the skip/xfail. If omitted, a default reason is used.
    exceptions : list[str], optional
        A list of exceptions for use with ``cpu_only``, ``eager_only``, or ``np_only``.
        This should be provided when delegation is implemented for some,
        but not all, non-CPU/non-NumPy backends.
    """
    if f"{skip_or_xfail}_xp_backends" not in request.keywords:
        return

    skip_xfail_reasons = _backends_kwargs_from_request(
        request, skip_or_xfail=skip_or_xfail
    )
    xp = request.param
    if xp.__name__ in skip_xfail_reasons:
        reason = skip_xfail_reasons[xp.__name__]
        assert reason  # Default reason applied above
        skip_or_xfail = getattr(pytest, skip_or_xfail)
        skip_or_xfail(reason=reason)

