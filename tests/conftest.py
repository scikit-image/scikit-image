# Fixtures for doctests
from pathlib import Path
import pytest
from time import perf_counter
from datetime import timedelta
import os
import sys
from pytest_pretty import CustomTerminalReporter

from _pytest.pathlib import bestrelpath


@pytest.fixture(autouse=True)
def handle_np2():
    # TODO: remove when we require numpy >= 2
    try:
        import numpy as np

        np.set_printoptions(legacy="1.21")
    except ImportError:
        pass


class SKTerminalReporter(CustomTerminalReporter):
    """
    This ia s custom terminal reporter to how long it takes to finish given part of tests.
    It prints time each time when test from different file is started.

    It is created to be able to see if timeout is caused by long time execution, or it is just hanging.
    """

    currentfspath: Path | None
    _start_time: float | None

    def write_fspath_result(self, nodeid: str, res, **markup: bool) -> None:
        if getattr(self, '_start_time', None) is None:
            self._start_time = perf_counter()
        fspath = self.config.rootpath / nodeid.split('::')[0]
        if self.currentfspath is None or fspath != self.currentfspath:
            if self.currentfspath is not None and self._show_progress_info:
                self._write_progress_information_filling_space()
                if os.environ.get('CI', False):
                    self.write(
                        f' [{timedelta(seconds=int(perf_counter() - self._start_time))}]'
                    )
            self.currentfspath = fspath
            relfspath = bestrelpath(self.startpath, fspath)
            self._tw.line()
            self.write(relfspath + ' ')
        self.write(res, flush=True, **markup)


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    # Get the standard terminal reporter plugin and replace it with our
    standard_reporter = config.pluginmanager.getplugin('terminalreporter')
    custom_reporter = SKTerminalReporter(config, sys.stdout)
    if standard_reporter._session is not None:
        custom_reporter._session = standard_reporter._session
    config.pluginmanager.unregister(standard_reporter)
    config.pluginmanager.register(custom_reporter, 'terminalreporter')
