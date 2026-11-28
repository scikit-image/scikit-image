# Fixtures for doctests
from pathlib import Path
import pytest
from time import perf_counter
from datetime import timedelta
import os
import sys
import sysconfig
from importlib.util import find_spec
from pytest_pretty import CustomTerminalReporter

from _pytest.terminal import TerminalReporter
from _pytest.pathlib import bestrelpath

import numpy as np

from . import gaussian_sum


# Builds made with `-Dexclude-v2=true` do not ship `skimage2`.
collect_ignore_glob = [] if find_spec("skimage2") else ["skimage2/*"]

FREE_THREADED_BUILD = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
GIL_ENABLED_AT_START = getattr(sys, "_is_gil_enabled", lambda: True)()

try:
    import pytest_run_parallel  # noqa:F401

    PARALLEL_RUN_AVAILABLE = True
except ImportError:
    PARALLEL_RUN_AVAILABLE = False


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
    if not PARALLEL_RUN_AVAILABLE:
        config.addinivalue_line(
            "markers",
            "thread_unsafe: mark the test function as single-threaded",
        )


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


@pytest.fixture(scope='session')
def gaussian_sum_images():
    """``{name: (levels, image)}``: Gaussian sums fitted to three photographs.

    ``levels`` are the `gaussian_sum.Level` objects of the fit, and ``image``
    is the fitted sum at ``σ = 0``. Use `gaussian_sum.render` and
    `gaussian_sum.hessian` for its exact smoothed values and derivatives.

    The fits are checked to stay within 10% RMS of the photographs. A worse
    fit would not fail tests that take their standard from the same fit, but
    it would make the test images less like photographs.
    """
    from _skimage2 import data, img_as_float
    from _skimage2.color import rgb2gray

    photographs = {
        'camera': img_as_float(data.camera()),
        'coins': img_as_float(data.coins()),
        'astronaut': img_as_float(rgb2gray(data.astronaut())),
    }
    fitted = {}
    for name, photograph in photographs.items():
        levels, residual = gaussian_sum.fit(photograph)
        relative_rms = np.sqrt(np.mean(residual**2) / np.mean(photograph**2))
        assert relative_rms < 0.10, (
            f'{name}: fit RMS residual is {relative_rms:.1%} of image RMS'
        )
        fitted[name] = (levels, gaussian_sum.render(levels, photograph.shape))
    return fitted
