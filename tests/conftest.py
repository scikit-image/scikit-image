# Fixtures for doctests
from pathlib import Path
import pytest
from time import perf_counter
from datetime import timedelta
import os
import sys
from pytest_pretty import CustomTerminalReporter

from _pytest.pathlib import bestrelpath


class SKTerminalReporter(CustomTerminalReporter):
    """
    Custom terminal reporter to display test runtimes.
    It displays the cumulative runtime each time a new file is tested.
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
    # Get the standard terminal reporter plugin and replace it with ours
    standard_reporter = config.pluginmanager.getplugin('terminalreporter')
    custom_reporter = SKTerminalReporter(config, sys.stdout)
    if standard_reporter._session is not None:
        # if session is already set we need to copy them from
        # the previous reporter
        custom_reporter._session = standard_reporter._session
    config.pluginmanager.unregister(standard_reporter)
    config.pluginmanager.register(custom_reporter, 'terminalreporter')
