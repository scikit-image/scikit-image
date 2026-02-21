from __future__ import annotations as _annotations

import re
import shutil
import sys
from importlib.metadata import version as _metadata_version
from itertools import dropwhile
from time import perf_counter_ns
from typing import TYPE_CHECKING, Callable

import pytest
from _pytest.terminal import TerminalReporter
from rich.console import Console
from rich.markup import escape
from rich.table import Table

if TYPE_CHECKING:
    from _pytest.reports import TestReport

    SummaryStats = tuple[list[tuple[str, dict[str, bool]]], str]

__version__ = _metadata_version('pytest-pretty')
start_time = 0
end_time = 0
console = Console(width=shutil.get_terminal_size()[0])


def pytest_sessionstart(session):
    global start_time
    start_time = perf_counter_ns()


def pytest_sessionfinish(session, exitstatus):
    global end_time
    end_time = perf_counter_ns()


class CustomTerminalReporter(TerminalReporter):
    def pytest_runtest_logreport(self, report: TestReport) -> None:
        super().pytest_runtest_logreport(report)
        if not report.failed:
            return None
        if report.when == 'teardown' and 'devtools-insert-assert:' in repr(report.longrepr):
            # special case for devtools insert_assert "failures"
            return None
        file, line, func = report.location
        self._write_progress_information_filling_space()
        self.ensure_newline()
        summary = f'{file}:{line} {func}'
        self._tw.write(summary, red=True)
        try:
            msg = report.longrepr.reprcrash.message
        except AttributeError:
            pass
        else:
            msg = msg.replace('\n', ' ')
            available_space = self._tw.fullwidth - len(summary) - 15
            if available_space > 5:
                self._tw.write(f' - {msg[:available_space]}…')

    def summary_stats(self) -> None:
        time_taken_ns = end_time - start_time
        summary_items, _ = self.build_summary_stats_line()
        console.print(f'[bold]Results ({time_taken_ns / 1_000_000_000:0.2f}s):[/]', highlight=False)
        for summary_item in summary_items:
            msg, text_format = summary_item
            text_format.pop('bold', None)
            color = next(k for k, v in text_format.items() if v)
            count, label = msg.split(' ', 1)
            console.print(f'{count:>10} {label}', style=color)

    def short_test_summary(self) -> None:
        fail_reports = self.stats.get('failed', [])
        if fail_reports:
            table = Table(title='Summary of Failures', padding=(0, 2), border_style='cyan')
            table.add_column('File')
            table.add_column('Function', style='bold')
            table.add_column('Function Line', style='bold')
            table.add_column('Error Line')
            table.add_column('Error')
            for report in fail_reports:
                file, function_line, func = report.location
                try:
                    repr_entries = report.longrepr.chain[-1][0].reprentries
                    error_line = str(repr_entries[0].reprfileloc.lineno)
                    error = repr_entries[-1].reprfileloc.message
                except AttributeError:
                    error_line = ''
                    error = ''

                table.add_row(
                    escape(file),
                    escape(func),
                    str(function_line + 1) if function_line is not None else '',
                    escape(error_line),
                    escape(error),
                )
            console.print(table)


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    # Get the standard terminal reporter plugin and replace it with our
    standard_reporter = config.pluginmanager.getplugin('terminalreporter')
    custom_reporter = CustomTerminalReporter(config, sys.stdout)
    config.pluginmanager.unregister(standard_reporter)
    config.pluginmanager.register(custom_reporter, 'terminalreporter')


ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
stat_re = re.compile(r'(\d+) (\w+)')


def create_new_parseoutcomes(runresult_instance) -> Callable[[], dict[str, int]]:
    """
    In this function there is a new implementation of `RunResult.parseoutcomes`
    https://github.com/pytest-dev/pytest/blob/4a46ee8bc957b06265c016cc837862447dde79d2/src/_pytest/pytester.py#L557


    Decision of reimplement this method is made based on implementation of
    `RunResult.assert_outcomes`

    https://github.com/pytest-dev/pytest/blob/4a46ee8bc957b06265c016cc837862447dde79d2/src/_pytest/pytester.py#L613
    """

    def parseoutcomes() -> dict[str, int]:
        lines_with_stats = dropwhile(lambda x: 'Results' not in x, runresult_instance.outlines)
        next(lines_with_stats)  # drop Results line
        res = {}
        for i, line in enumerate(lines_with_stats):
            line = ansi_escape.sub('', line).strip()  # clean colors
            match = stat_re.match(line)

            if match is None:
                break

            res[match.group(2)] = int(match.group(1))

        return res

    return parseoutcomes


class PytesterWrapper:
    """
    This is class for for make almost transparent wrapper
    around pytester output and allow substitute
    `parseoutcomes` method of `RunResult` instance.
    """

    __slot__ = ('_pytester',)

    def __init__(self, pytester):
        object.__setattr__(self, '_pytester', pytester)

    def runpytest(self, *args, **kwargs):
        """wrapper to overwrite `parseoutcomes` method of `RunResult` instance"""
        res = self._pytester.runpytest(*args, **kwargs)
        assert res is not None
        res.parseoutcomes = create_new_parseoutcomes(res)
        return res

    def __getattr__(self, name):
        return getattr(self._pytester, name)

    def __setattr__(self, name, value):
        setattr(self._pytester, name, value)


@pytest.fixture()
def pytester_pretty(pytester):
    return PytesterWrapper(pytester)
