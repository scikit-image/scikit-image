# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This plugin provides advanced doctest support and enables the testing of .rst
files.
"""
import doctest
import fnmatch
import os
import re
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path
import subprocess
from textwrap import indent
from unittest import SkipTest

import pytest
from _pytest.outcomes import OutcomeException  # Private API, but been around since 3.7
from _pytest.doctest import _get_continue_on_failure  # Since 3.5, still in 7.3
from packaging.version import Version

from pytest_doctestplus.utils import ModuleChecker

from .output_checker import (FIX, IGNORE_WARNINGS, REMOTE_DATA, SHOW_WARNINGS,
                             OutputChecker)

_pytest_version = Version(pytest.__version__)
PYTEST_GE_8_0 = _pytest_version >= Version('8.0')
PYTEST_GE_8_1_1 = _pytest_version >= Version('8.1.1')

comment_characters = {
    '.txt': '#',
    '.tex': '%',
    '.rst': r'\.\.',
    '.md': '<!--'
}


# For the IGNORE_WARNINGS and SHOW_WARNINGS option, we create a context manager
# that doesn't require us to add any imports to the example list and contains
# everything that is needed to silence or print warnings.

IGNORE_WARNINGS_CONTEXT = """
class _doctestplus_ignore_all_warnings:

    def __init__(self):
        import warnings
        self._cw = warnings.catch_warnings()

    def __enter__(self, *args, **kwargs):
        result = self._cw.__enter__(*args, **kwargs)
        import warnings
        warnings.simplefilter('ignore')
        return result

    def __exit__(self, *args, **kwargs):
        return self._cw.__exit__(*args, **kwargs)
""".lstrip()


SHOW_WARNINGS_CONTEXT = """
class _doctestplus_show_all_warnings:

    def __init__(self):
        import warnings
        self._cw = warnings.catch_warnings(record=True)

    def __enter__(self, *args, **kwargs):
        self.result = self._cw.__enter__(*args, **kwargs)
        import warnings
        warnings.simplefilter('always')
        return self.result

    def __exit__(self, *args, **kwargs):
        self._cw.__exit__(*args, **kwargs)
        for warn in self.result:
            print(f'{warn._category_name}: {warn.message}')
""".lstrip()


# these pytest hooks allow us to mark tests and run the marked tests with
# specific command line options.
def pytest_addoption(parser):
    parser.addoption("--doctest-plus", action="store_true",
                     help="enable running doctests with additional "
                          "features not found in the normal doctest "
                          "plugin")

    parser.addoption("--doctest-ufunc", action="store_true",
                     help="enable running doctests in docstrings of Numpy "
                          "ufuncs")

    parser.addoption("--doctest-rst", action="store_true",
                     help=(
                         "Enable running doctests in .rst documentation. "
                         "This is no longer recommended, use --doctest-glob instead."
                     ))

    parser.addoption("--text-file-format", action="store",
                     help=(
                         "Text file format for narrative documentation. "
                         "Options accepted are 'txt', 'tex', and 'rst'. "
                         "This is no longer recommended, use --doctest-glob instead."
                     ))

    # Defaults to `atol` parameter from `numpy.allclose`.
    parser.addoption("--doctest-plus-atol", action="store",
                     help="set the absolute tolerance for float comparison",
                     default=1e-08)

    # Defaults to `rtol` parameter from `numpy.allclose`.
    parser.addoption("--doctest-plus-rtol", action="store",
                     help="set the relative tolerance for float comparison",
                     default=1e-05)

    parser.addoption("--doctest-only", action="store_true",
                     help="Test only doctests. Implies usage of doctest-plus.")

    parser.addoption("--doctest-plus-generate-diff",
                     help=(
                         "Generate a diff where expected output and actual "
                         "output differ.  "
                         "The diff is printed to stdout if not using "
                         "`--doctest-plus-generate-diff=overwrite` which "
                         "causes editing of the original files.\n"
                         "NOTE: Unless an in-pace build is picked up, python "
                         "file paths may point to unexpected places. "
                         "If 'overwrite' is not used, will create a temporary "
                         "folder and use `git diff -p` to generate a diff."),
                     choices=["diff", "overwrite"],
                     action="store", nargs="?", default=False, const="diff")

    parser.addini("text_file_format",
                  "Default format for docs. "
                  "This is no longer recommended, use --doctest-glob instead.")

    parser.addini("doctest_optionflags", "option flags for doctests",
                  type="args", default=["ELLIPSIS", "NORMALIZE_WHITESPACE"],)

    parser.addini("doctest_plus", "enable running doctests with additional "
                                  "features not found in the normal doctest plugin")

    parser.addini("doctest_ufunc", "enable running doctests in docstrings of "
                                   "Numpy ufuncs")

    parser.addini("doctest_norecursedirs",
                  "like the norecursedirs option but applies only to doctest "
                  "collection", type="args", default=())

    parser.addini("doctest_rst",
                  "Run the doctests in the rst documentation",
                  default=False)

    parser.addini("doctest_plus_atol",
                  "set the absolute tolerance for float comparison",
                  default=1e-08)

    parser.addini("doctest_plus_rtol",
                  "set the relative tolerance for float comparison",
                  default=1e-05)

    parser.addini('text_file_comment_chars',
                  help='list of pairs in format file_extension=comment_chars, eg: .rst=..',
                  type='linelist',
                  default=[])

    parser.addini("doctest_subpackage_requires",
                  "A list of paths to skip if requirements are not satisfied."
                  "Each item in the list should have the syntax path=req1;req2",
                  type='linelist',
                  default=[])


def pytest_addhooks(pluginmanager):
    from pytest_doctestplus import newhooks
    pluginmanager.add_hookspecs(newhooks)


def get_optionflags(parent):
    optionflags_str = parent.config.getini('doctest_optionflags')
    flag_int = 0
    for flag_str in optionflags_str:
        flag_int |= doctest.OPTIONFLAGS_BY_NAME[flag_str]
    return flag_int


def _is_numpy_ufunc(method):
    try:
        import numpy as np
    except ModuleNotFoundError:
        # If Numpy is not installed, then there can't be any ufuncs!
        return False
    while True:
        try:
            method = method.__wrapped__
        except AttributeError:
            break
    return isinstance(method, np.ufunc)


def pytest_configure(config):
    doctest_plugin = config.pluginmanager.getplugin("doctest")
    if not hasattr(config.option, "doctestmodules"):
        return
    run_regular_doctest = (
        config.option.doctestmodules and not config.option.doctest_plus
    )
    if config.option.doctest_plus_generate_diff:
        config.option.doctest_only = True
    use_doctest_plus = config.getini(
        'doctest_plus') or config.option.doctest_plus or config.option.doctest_only
    use_doctest_ufunc = config.getini(
        'doctest_ufunc') or config.option.doctest_ufunc
    if doctest_plugin is None or run_regular_doctest or not use_doctest_plus:
        return

    # We monkey-patch in our replacement doctest OutputChecker.  Not
    # great, but there isn't really an API to replace the checker when
    # using doctest.testfile, unfortunately.
    doctest.OutputChecker = OutputChecker
    OutputChecker.rtol = max(float(config.getini("doctest_plus_rtol")),
                             float(config.getoption("doctest_plus_rtol")))
    OutputChecker.atol = max(float(config.getini("doctest_plus_atol")),
                             float(config.getoption("doctest_plus_atol")))

    use_rst = config.getini('doctest_rst') or config.option.doctest_rst
    file_ext = config.option.text_file_format or config.getini('text_file_format') or 'rst'
    if use_rst:
        config.option.doctestglob.append(f'*.{file_ext}')

    # override default comment characters
    ext_comment_pairs = [pair.split('=') for pair in config.getini('text_file_comment_chars')]
    for ext, chars in ext_comment_pairs:
        comment_characters[ext] = chars

    # Fetch the global hook function:
    global doctestplus_diffhook
    doctestplus_diffhook = config.hook.pytest_doctestplus_diffhook

    class DocTestModulePlus(doctest_plugin.DoctestModule):
        # pytest 2.4.0 defines "collect".  Prior to that, it defined
        # "runtest".  The "collect" approach is better, because we can
        # skip modules altogether that have no doctests.  However, we
        # need to continue to override "runtest" so that the built-in
        # behavior (which doesn't do whitespace normalization or
        # handling __doctest_skip__) doesn't happen.
        def collect(self):
            # When running directly from pytest we need to make sure that we
            # don't accidentally import setup.py!
            fspath = self.path
            filepath = self.path.name

            if filepath in ("setup.py", "__main__.py"):
                return
            try:
                from _pytest.pathlib import import_path
                mode = self.config.getoption("importmode")

                if PYTEST_GE_8_1_1:
                    consider_namespace_packages = self.config.getini("consider_namespace_packages")
                    module = import_path(fspath, mode=mode, root=self.config.rootpath,
                                         consider_namespace_packages=consider_namespace_packages)
                else:
                    module = import_path(fspath, mode=mode, root=self.config.rootpath)
            except ImportError:
                if self.config.getvalue("doctest_ignore_import_errors"):
                    pytest.skip("unable to import module %r" % fspath)
                else:
                    raise

            options = get_optionflags(self) | FIX

            # uses internal doctest module parsing mechanism
            finder = DocTestFinderPlus(doctest_ufunc=use_doctest_ufunc)
            runner = DebugRunnerPlus(
                verbose=False,
                optionflags=options,
                checker=OutputChecker(),
                # Helper disables continue-on-failure when debugging is enabled
                continue_on_failure=_get_continue_on_failure(config),
                generate_diff=config.option.doctest_plus_generate_diff,
            )

            for test in finder.find(module):
                if test.examples:  # skip empty doctests
                    ignore_warnings_context_needed = False
                    show_warnings_context_needed = False

                    for example in test.examples:
                        if (config.getoption('remote_data', 'none') != 'any'
                                and example.options.get(REMOTE_DATA)):
                            example.options[doctest.SKIP] = True

                        # If warnings are to be ignored we need to catch them by
                        # wrapping the source in a context manager.
                        elif example.options.get(IGNORE_WARNINGS, False):
                            example.source = ("with _doctestplus_ignore_all_warnings():\n"
                                              + indent(example.source, '    '))
                            ignore_warnings_context_needed = True

                        # Same for SHOW_WARNINGS
                        elif example.options.get(SHOW_WARNINGS, False):
                            example.source = ("with _doctestplus_show_all_warnings():\n"
                                              + indent(example.source, '    '))
                            show_warnings_context_needed = True

                    # We insert the definition of the context manager to ignore
                    # warnings at the start of the file if needed.
                    if ignore_warnings_context_needed:
                        test.examples.insert(0, doctest.Example(
                            source=IGNORE_WARNINGS_CONTEXT, want=''))

                    if show_warnings_context_needed:
                        test.examples.insert(0, doctest.Example(
                            source=SHOW_WARNINGS_CONTEXT, want=''))

                    try:
                        yield doctest_plugin.DoctestItem.from_parent(
                            self, name=test.name, runner=runner, dtest=test
                        )
                    except AttributeError:
                        # pytest < 5.4
                        yield doctest_plugin.DoctestItem(
                            test.name, self, runner, test)

    class DocTestTextfilePlus(pytest.Module):
        obj = None

        def collect(self):
            fspath = self.path
            filepath = self.path.name

            encoding = self.config.getini("doctest_encoding")
            text = fspath.read_text(encoding)
            filename = str(fspath)
            globs = {"__name__": "__main__"}

            optionflags = get_optionflags(self) | FIX

            runner = DebugRunnerPlus(
                verbose=False, optionflags=optionflags, checker=OutputChecker(),
                continue_on_failure=_get_continue_on_failure(self.config),
                generate_diff=self.config.option.doctest_plus_generate_diff,
            )

            parser = DocTestParserPlus()
            test = parser.get_doctest(text, globs, filepath, filename, 0)
            if test.examples:
                try:
                    yield doctest_plugin.DoctestItem.from_parent(
                        self, name=test.name, runner=runner, dtest=test
                    )
                except AttributeError:
                    # pytest < 5.4
                    yield doctest_plugin.DoctestItem(test.name, self, runner, test)

    class DocTestParserPlus(doctest.DocTestParser):
        """
        An extension to the builtin DocTestParser that handles the
        special directives for skipping tests.

        The directives are:

           - ``.. doctest-skip::``: Skip the next doctest chunk.

           - ``.. doctest-requires:: module1, module2``: Skip the next
             doctest chunk if the given modules/packages are not
             installed.

           - ``.. doctest-requires-all:: module1, module2``: Skip all subsequent
             doctest chunks if the given modules/packages are not
             installed.

           - ``.. doctest-skip-all``: Skip all subsequent doctests.

           - ``.. doctest-remote-data::``: Skip the next doctest chunk if
             --remote-data is not passed.

           - ``.. doctest-remote-data-all::``: Skip all subsequent doctest
             chunks if --remote-data is not passed.
        """

        def parse(self, s, name=None):
            result = doctest.DocTestParser.parse(self, s, name=name)

            # result is a sequence of alternating text chunks and
            # doctest.Example objects.  We need to look in the text
            # chunks for the special directives that help us determine
            # whether the following examples should be skipped.

            required = []
            skip_next = False
            skip_all = False

            ext = os.path.splitext(name)[1] if name else '.rst'
            if ext not in comment_characters:
                warnings.warn("file format '{}' is not recognized, assuming "
                              "'{}' as the comment character."
                              .format(ext, comment_characters['.rst']))
                ext = '.rst'
            comment_char = comment_characters[ext]

            ignore_warnings_context_needed = False
            show_warnings_context_needed = False

            for entry in result:

                if isinstance(entry, str) and entry:
                    required = []
                    required_all = []
                    skip_next = False
                    lines = entry.strip().splitlines()

                    requires_all_match = [re.match(
                        fr'{comment_char}\s+doctest-requires-all\s*::\s+(.*)', x) for x in lines]
                    if any(requires_all_match):
                        required_all = [re.split(r'\s*[,\s]\s*', match.group(1))
                                        for match in requires_all_match if match][0]
                    required_modules_all = DocTestFinderPlus.check_required_modules(required_all)
                    if not required_modules_all:
                        skip_all = True
                        continue

                    if config.getoption('remote_data', 'none') != 'any':
                        if any(re.match(fr'{comment_char}\s+doctest-remote-data-all\s*::', x.strip())  # noqa: E501
                               for x in lines):
                            skip_all = True
                            continue

                    if any(re.match(f'{comment_char} doctest-skip-all', x.strip()) for x in lines):
                        skip_all = True
                        continue

                    if not len(lines):
                        continue

                    # We allow last and second to last lines to match to allow
                    # special environment to be in between, e.g. \begin{python}
                    last_lines = lines[-2:]
                    matches = [re.match(
                        fr'{comment_char}\s+doctest-skip\s*::(\s+.*)?',
                        last_line) for last_line in last_lines]

                    if len(matches) > 1:
                        match = matches[0] or matches[1]
                    else:
                        match = matches[0]

                    if match:
                        marker = match.group(1)
                        if (marker is None or
                                (marker.strip() == 'win32' and
                                 sys.platform == 'win32')):
                            skip_next = True
                            continue

                    if config.getoption('remote_data', 'none') != 'any':
                        matches = (re.match(
                            fr'{comment_char}\s+doctest-remote-data\s*::',
                            last_line) for last_line in last_lines)

                        if any(matches):
                            skip_next = True
                            continue

                    matches = [re.match(
                        fr'{comment_char}\s+doctest-requires\s*::\s+(.*)',
                        last_line) for last_line in last_lines]

                    if len(matches) > 1:
                        match = matches[0] or matches[1]
                    else:
                        match = matches[0]

                    if match:
                        # 'a a' or 'a,a' or 'a, a'-> [a, a]
                        required = re.split(r'\s*[,\s]\s*', match.group(1))
                elif isinstance(entry, doctest.Example):

                    has_required_modules = DocTestFinderPlus.check_required_modules(required)
                    if skip_all or skip_next or not has_required_modules:
                        entry.options[doctest.SKIP] = True

                    elif (config.getoption('remote_data', 'none') != 'any'
                            and entry.options.get(REMOTE_DATA)):
                        entry.options[doctest.SKIP] = True

                    # If warnings are to be ignored we need to catch them by
                    # wrapping the source in a context manager.
                    elif entry.options.get(IGNORE_WARNINGS, False):
                        entry.source = ("with _doctestplus_ignore_all_warnings():\n"
                                        + indent(entry.source, '    '))
                        ignore_warnings_context_needed = True

                    # Same to show warnings
                    elif entry.options.get(SHOW_WARNINGS, False):
                        entry.source = ("with _doctestplus_show_all_warnings():\n"
                                        + indent(entry.source, '    '))
                        show_warnings_context_needed = True

            # We insert the definition of the context manager to ignore
            # warnings at the start of the file if needed.
            if ignore_warnings_context_needed:
                result.insert(0, doctest.Example(source=IGNORE_WARNINGS_CONTEXT, want=''))

            if show_warnings_context_needed:
                result.insert(0, doctest.Example(source=SHOW_WARNINGS_CONTEXT, want=''))

            return result

    config.pluginmanager.register(
        DoctestPlus(
            DocTestModulePlus,
            DocTestTextfilePlus,
            config.option.doctestglob,
        ),
        'doctestplus',
    )
    # Remove the doctest_plugin, or we'll end up testing the .rst files twice.
    config.pluginmanager.unregister(doctest_plugin)


class DoctestPlus:
    def __init__(self, doctest_module_item_cls, doctest_textfile_item_cls, file_globs):
        """
        doctest_module_item_cls should be a class inheriting
        `pytest.doctest.DoctestItem` and `pytest.File`.  This class handles
        running of a single doctest found in a Python module.  This is passed
        in as an argument because the actual class to be used may not be
        available at import time, depending on whether or not the doctest
        plugin for py.test is available.
        """
        self._doctest_module_item_cls = doctest_module_item_cls
        self._doctest_textfile_item_cls = doctest_textfile_item_cls
        self._file_globs = file_globs
        # Directories to ignore when adding doctests
        self._ignore_paths = []

    if PYTEST_GE_8_0:

        def pytest_ignore_collect(self, collection_path, config):
            """
            Skip paths that match any of the doctest_norecursedirs patterns or
            if doctest_only is True then skip all regular test files (eg test_*.py).
            """
            from _pytest.pathlib import fnmatch_ex

            collect_ignore = config._getconftest_pathlist("collect_ignore",
                                                          path=collection_path.parent)

            # The collect_ignore conftest.py variable should cause all test
            # runners to ignore this file and all subfiles and subdirectories
            if collect_ignore is not None and collection_path in collect_ignore:
                return True

            if config.option.doctest_only:
                for pattern in config.getini('python_files'):
                    if fnmatch_ex(pattern, collection_path):
                        return True

            def get_list_opt(name):
                return getattr(config.option, name, None) or []

            for ignore_path in get_list_opt('ignore'):
                ignore_path = os.path.abspath(ignore_path)
                if str(collection_path).startswith(ignore_path):
                    return True

            for pattern in get_list_opt('ignore_glob'):
                if fnmatch_ex(pattern, collection_path):
                    return True

            for pattern in config.getini("doctest_norecursedirs"):
                if fnmatch_ex(pattern, collection_path):
                    # Apparently pytest_ignore_collect causes files not to be
                    # collected by any test runner; for DoctestPlus we only want to
                    # avoid creating doctest nodes for them
                    self._ignore_paths.append(collection_path)
                    break

            for option in config.getini("doctest_subpackage_requires"):
                subpackage_pattern, required = option.split('=', 1)
                if fnmatch_ex(subpackage_pattern.strip(), collection_path):
                    required = required.strip().split(';')
                    if not DocTestFinderPlus.check_required_modules(required):
                        self._ignore_paths.append(collection_path)
                        break

            # Let other plugins decide the outcome.
            return None

        def pytest_collect_file(self, file_path, parent):
            """Implements an enhanced version of the doctest module from py.test
            (specifically, as enabled by the --doctest-modules option) which
            supports skipping all doctests in a specific docstring by way of a
            special ``__doctest_skip__`` module-level variable.  It can also skip
            tests that have special requirements by way of
            ``__doctest_requires__``.

            ``__doctest_skip__`` should be a list of functions, classes, or class
            methods whose docstrings should be ignored when collecting doctests.

            This also supports wildcard patterns.  For example, to run doctests in
            a class's docstring, but skip all doctests in its modules use, at the
            module level::

                __doctest_skip__ = ['ClassName.*']

            You may also use the string ``'.'`` in ``__doctest_skip__`` to refer
            to the module itself, in case its module-level docstring contains
            doctests.

            ``__doctest_requires__`` should be a dictionary mapping wildcard
            patterns (in the same format as ``__doctest_skip__``) to a list of one
            or more modules that should be *importable* in order for the tests to
            run.  For example, if some tests require the scipy module to work they
            will be skipped unless ``import scipy`` is possible.  It is also
            possible to use a tuple of wildcard patterns as a key in this dict::

                __doctest_requires__ = {('func1', 'func2'): ['scipy']}

            """
            from _pytest.pathlib import commonpath, fnmatch_ex

            for ignore_path in self._ignore_paths:
                if commonpath(ignore_path, file_path) == ignore_path:
                    return None

            if file_path.suffix == '.py':
                if file_path.name == 'conf.py':
                    return None

                # Don't override the built-in doctest plugin
                return self._doctest_module_item_cls.from_parent(parent, path=file_path)

            elif any([fnmatch_ex(pat, file_path) for pat in self._file_globs]):
                # Ignore generated .rst files
                parts = str(file_path).split(os.path.sep)

                # Don't test files that start with a _
                if file_path.name.startswith('_'):
                    return None

                # Don't test files in directories that start with a '_' if those
                # directories are inside docs. Note that we *should* allow for
                # example /tmp/_q/docs/file.rst but not /tmp/docs/_build/file.rst
                # If we don't find 'docs' in the path, we should just skip this
                # check to be safe. We also want to skip any api sub-directory
                # of docs.
                if 'docs' in parts:
                    # We index from the end on the off chance that the temporary
                    # directory includes 'docs' in the path, e.g.
                    # /tmp/docs/371j/docs/index.rst You laugh, but who knows! :)
                    # Also, it turns out lists don't have an rindex method. Huh??!!
                    docs_index = len(parts) - 1 - parts[::-1].index('docs')
                    if any(x.startswith('_') or x == 'api' for x in parts[docs_index:]):
                        return None

                # TODO: Get better names on these items when they are
                # displayed in py.test output
                return self._doctest_textfile_item_cls.from_parent(parent, path=file_path)

    else:  # PYTEST_LT_8

        def pytest_ignore_collect(self, path, config):
            """
            Skip paths that match any of the doctest_norecursedirs patterns or
            if doctest_only is True then skip all regular test files (eg test_*.py).
            """
            dirpath = Path(path).parent
            collect_ignore = config._getconftest_pathlist("collect_ignore",
                                                          path=dirpath,
                                                          rootpath=config.rootpath)

            # The collect_ignore conftest.py variable should cause all test
            # runners to ignore this file and all subfiles and subdirectories
            if collect_ignore is not None and path in collect_ignore:
                return True

            if config.option.doctest_only:
                for pattern in config.getini('python_files'):
                    if path.check(fnmatch=pattern):
                        return True

            def get_list_opt(name):
                return getattr(config.option, name, None) or []

            for ignore_path in get_list_opt('ignore'):
                ignore_path = os.path.abspath(ignore_path)
                if str(path).startswith(ignore_path):
                    return True

            for pattern in get_list_opt('ignore_glob'):
                if path.check(fnmatch=pattern):
                    return True

            for pattern in config.getini("doctest_norecursedirs"):
                if path.check(fnmatch=pattern):
                    # Apparently pytest_ignore_collect causes files not to be
                    # collected by any test runner; for DoctestPlus we only want to
                    # avoid creating doctest nodes for them
                    self._ignore_paths.append(path)
                    break

            for option in config.getini("doctest_subpackage_requires"):
                subpackage_pattern, required = option.split('=', 1)
                if path.check(fnmatch=subpackage_pattern.strip()):
                    required = required.strip().split(';')
                    if not DocTestFinderPlus.check_required_modules(required):
                        self._ignore_paths.append(path)
                        break

            # Let other plugins decide the outcome.
            return None

        def pytest_collect_file(self, path, parent):
            """Implements an enhanced version of the doctest module from py.test
            (specifically, as enabled by the --doctest-modules option) which
            supports skipping all doctests in a specific docstring by way of a
            special ``__doctest_skip__`` module-level variable.  It can also skip
            tests that have special requirements by way of
            ``__doctest_requires__``.

            ``__doctest_skip__`` should be a list of functions, classes, or class
            methods whose docstrings should be ignored when collecting doctests.

            This also supports wildcard patterns.  For example, to run doctests in
            a class's docstring, but skip all doctests in its modules use, at the
            module level::

                __doctest_skip__ = ['ClassName.*']

            You may also use the string ``'.'`` in ``__doctest_skip__`` to refer
            to the module itself, in case its module-level docstring contains
            doctests.

            ``__doctest_requires__`` should be a dictionary mapping wildcard
            patterns (in the same format as ``__doctest_skip__``) to a list of one
            or more modules that should be *importable* in order for the tests to
            run.  For example, if some tests require the scipy module to work they
            will be skipped unless ``import scipy`` is possible.  It is also
            possible to use a tuple of wildcard patterns as a key in this dict::

                __doctest_requires__ = {('func1', 'func2'): ['scipy']}

            """
            for ignore_path in self._ignore_paths:
                if ignore_path.common(path) == ignore_path:
                    return None

            if path.ext == '.py':
                if path.basename == 'conf.py':
                    return None

                # Don't override the built-in doctest plugin
                return self._doctest_module_item_cls.from_parent(parent, path=Path(path))

            elif any([path.check(fnmatch=pat) for pat in self._file_globs]):
                # Ignore generated .rst files
                parts = str(path).split(os.path.sep)

                # Don't test files that start with a _
                if path.basename.startswith('_'):
                    return None

                # Don't test files in directories that start with a '_' if those
                # directories are inside docs. Note that we *should* allow for
                # example /tmp/_q/docs/file.rst but not /tmp/docs/_build/file.rst
                # If we don't find 'docs' in the path, we should just skip this
                # check to be safe. We also want to skip any api sub-directory
                # of docs.
                if 'docs' in parts:
                    # We index from the end on the off chance that the temporary
                    # directory includes 'docs' in the path, e.g.
                    # /tmp/docs/371j/docs/index.rst You laugh, but who knows! :)
                    # Also, it turns out lists don't have an rindex method. Huh??!!
                    docs_index = len(parts) - 1 - parts[::-1].index('docs')
                    if any(x.startswith('_') or x == 'api' for x in parts[docs_index:]):
                        return None

                # TODO: Get better names on these items when they are
                # displayed in py.test output
                return self._doctest_textfile_item_cls.from_parent(parent, path=Path(path))


class DocTestFinderPlus(doctest.DocTestFinder):
    """Extension to the default `doctest.DoctestFinder` that supports
    ``__doctest_skip__`` magic.  See `pytest_collect_file` for more details.
    """

    # Caches the results of import attempts
    _import_cache = {}
    _module_checker = ModuleChecker()

    def __init__(self, *args, doctest_ufunc=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._doctest_ufunc = doctest_ufunc

    @classmethod
    def check_required_modules(cls, mods):
        """Check that modules in `mods` list are available.

        Parameters
        ----------
        mods : list of str
            List of modules. Modules can have specified versions (eg 'numpy>=1.15')

        Returns
        -------
        bool
            True if all modules in list are available.
        """
        for mod in mods:
            if mod in cls._import_cache:
                if not cls._import_cache[mod]:
                    return False
                continue

            if cls._module_checker.check(mod):
                cls._import_cache[mod] = True
            else:
                cls._import_cache[mod] = False
                return False
        return True

    def find(self, obj, name=None, module=None, globs=None, extraglobs=None):
        tests = doctest.DocTestFinder.find(self, obj, name, module, globs, extraglobs)

        if name is None and hasattr(obj, '__name__'):
            name = obj.__name__
        else:
            raise ValueError(
                "DocTestFinder.find: name must be given when obj.__name__ doesn't exist: "
                f"{type(obj)!r}"
            )

        if self._doctest_ufunc:
            for ufunc_name, ufunc_method in obj.__dict__.items():
                if _is_numpy_ufunc(ufunc_method):
                    tests += doctest.DocTestFinder.find(
                        self, ufunc_method, f'{name}.{ufunc_name}',
                        module=obj, globs=globs, extraglobs=extraglobs)

        if hasattr(obj, '__doctest_skip__') or hasattr(obj, '__doctest_requires__'):

            def conditionally_insert_skip(test):
                """
                Insert skip statement if `test` matches `__doctest_(skip|requires)__`.
                """
                for pat in getattr(obj, '__doctest_skip__', []):
                    if pat == '*':
                        self._prepend_skip(test)
                    elif pat == '.' and test.name == name:
                        self._prepend_skip(test)
                    elif fnmatch.fnmatch(test.name, '.'.join((name, pat))):
                        self._prepend_skip(test)

                reqs = getattr(obj, '__doctest_requires__', {})
                for pats, mods in reqs.items():
                    if not isinstance(pats, tuple):
                        pats = (pats,)

                    for pat in pats:
                        if pat == '*':
                            pass
                        elif pat == '.' and test.name == name:
                            pass
                        elif fnmatch.fnmatch(test.name, '.'.join((name, pat))):
                            pass
                        else:
                            continue  # The pattern does not apply

                        for mod in mods:
                            self._prepend_module_check(test, module=mod)
                return True

            for _test in tests:
                conditionally_insert_skip(_test)

        return tests

    def _prepend_skip(self, test):
        """Prepends `pytest.skip` before the doctest."""
        source = (
            "import pytest; "
            "pytest.skip('listed in `__doctest_skip__`'); "
            # Don't impact what's available in the namespace
            "del pytest"
        )
        importorskip = doctest.Example(source=source, want="")
        test.examples.insert(0, importorskip)

    def _prepend_module_check(self, test, *, module):
        """Prepends module checker before the doctest."""
        escaped_module = module.replace("'", "\\'")
        source = (
            "from pytest_doctestplus.utils import ModuleChecker; "
            "import pytest; "
            # Hide output of this statement in `___`, otherwise doctests fail
            f"___ = ModuleChecker().check('{escaped_module}') or "
            f"pytest.skip('could not import {escaped_module}'); "
            # Don't impact what's available in the namespace
            "del ModuleChecker, pytest, ___"
        )
        module_check = doctest.Example(source=source, want="")
        test.examples.insert(0, module_check)


def write_modified_file(fname, new_fname, changes, encoding=None):
    # Sort in reversed order to edit the lines:
    bad_tests = []
    changes.sort(key=lambda x: (x["test_lineno"], x["example_lineno"]),
                 reverse=True)

    with open(fname, encoding=encoding) as f:
        text = f.readlines()

    for change in changes:
        if change["test_lineno"] is None:
            bad_tests.append(change["name"])
            continue
        # Find the first line of the output:
        lineno = change["test_lineno"] + change["example_lineno"]
        lineno += change["source"].count("\n")

        indentation = len(text[lineno-1]) - len(text[lineno-1].lstrip())
        indentation = text[lineno-1][:indentation]
        want = indent(change["want"], indentation, lambda x: True)
        # Replace fully blank lines with the required `<BLANKLINE>`
        # (May need to do this also if line contains only whitespace)
        got = change["got"].replace("\n\n", "\n<BLANKLINE>\n")
        got = indent(got, indentation, lambda x: True)

        text[lineno:lineno+want.count("\n")] = [got]

    with open(new_fname, "w", encoding=encoding) as f:
        f.write("".join(text))

    return bad_tests


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    changesets = DebugRunnerPlus._changesets
    diff_mode = DebugRunnerPlus._generate_diff
    DebugRunnerPlus._changesets = defaultdict(list)
    DebugRunnerPlus._generate_diff = None
    all_bad_tests = []
    if not diff_mode:
        return  # we do not report or apply diffs

    encoding = config.getini("doctest_encoding")

    if diff_mode != "overwrite":
        # In this mode, we write a corrected file to a temporary folder in
        # order to compare them (rather than modifying the file).
        terminalreporter.section("Reporting DoctestPlus Diffs")
        if not changesets:
            terminalreporter.write_line("No doc changes to show")
            return

        # Strip away the common part of the path to make it a bit clearner...
        common_path = os.path.commonpath(changesets.keys())
        if not os.path.isdir(common_path):
            common_path = os.path.split(common_path)[0]

        with tempfile.TemporaryDirectory() as tmpdirname:
            for fname, changes in changesets.items():
                # Create a new filename and ensure the path exists (in the
                # temporary directory).
                new_fname = fname.replace(common_path, tmpdirname)
                os.makedirs(os.path.split(new_fname)[0], exist_ok=True)

                bad_tests = write_modified_file(fname, new_fname, changes, encoding)
                all_bad_tests.extend(bad_tests)

                # git diff returns 1 to signal changes, so just ignore the
                # exit status:
                with subprocess.Popen(
                        ["git", "diff", "-p", "--no-index", fname, new_fname],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding=encoding) as p:
                    p.wait()
                    # Diff should be fine, but write error if not:
                    diff = p.stderr.read()
                    diff += p.stdout.read()

                    # hide the temporary directory (cleaning up anyway):
                    if not os.path.isabs(common_path):
                        diff = diff.replace(tmpdirname, "/" + common_path)
                    else:
                        # diff seems to not include extra /
                        diff = diff.replace(tmpdirname, common_path)
                    terminalreporter.write(diff)
                    terminalreporter.write_line(f"{tmpdirname}, {common_path}")

                terminalreporter.section("Files with modifications", "-")
                terminalreporter.write_line(
                    "The following files would be overwritten with "
                    "`--doctest-plus-generate-diff=overwrite`:")
                for fname in changesets:
                    terminalreporter.write_line(f"    {fname}")
                terminalreporter.write_line(
                    "make sure these file paths are correct before calling it!")
    else:
        # We are in overwrite mode so will write the modified version directly
        # back into the same file and only report which files were changed.
        terminalreporter.section("DoctestPlus Fixing File Docs")
        if not changesets:
            terminalreporter.write_line("No doc changes to apply")
            return
        terminalreporter.write_line("Applied fix to the following files:")
        for fname, changes in changesets.items():
            bad_tests = write_modified_file(fname, fname, changes, encoding)
            all_bad_tests.extend(bad_tests)
            terminalreporter.write_line(f"    {fname}")

    if all_bad_tests:
        terminalreporter.section("Broken Linenumbers", "-")
        terminalreporter.write_line(
            "Doctestplus was unable to fix the following tests "
            "(their source is hidden or `__module__` overridden?)")
        for bad_test in all_bad_tests:
            terminalreporter.write_line(f"    {bad_test}")
        terminalreporter.write_line(
            "You can implementing a hook function to fix this (see README).")


class DebugRunnerPlus(doctest.DebugRunner):
    _changesets = defaultdict(list)
    _generate_diff = False

    def __init__(self, checker=None, verbose=None, optionflags=0,
                 continue_on_failure=True, generate_diff=False):
        # generated_diff is False, "diff", or "overwrite" (only need truthiness)
        DebugRunnerPlus._generate_diff = generate_diff

        super().__init__(checker=checker, verbose=verbose, optionflags=optionflags)
        self.continue_on_failure = continue_on_failure

    def report_success(self, out, test, example, got):
        if self._generate_diff:
            self.track_diff(False, out, test, example, got)
            return

        return super().report_success(out, test, example, got)

    def report_failure(self, out, test, example, got):
        if self._generate_diff:
            self.track_diff(True, out, test, example, got)
            return

        failure = doctest.DocTestFailure(test, example, got)
        if self.continue_on_failure:
            out.append(failure)
        else:
            raise failure

    def report_unexpected_exception(self, out, test, example, exc_info):
        cls, exception, traceback = exc_info
        if isinstance(exception, (OutcomeException, SkipTest)):
            raise exception
        failure = doctest.UnexpectedException(test, example, exc_info)
        if self.continue_on_failure:
            out.append(failure)
        else:
            raise failure

    def track_diff(self, use, out, test, example, got):
        if example.want == got:
            return

        info = dict(use=use, name=test.name, filename=test.filename,
                    source=example.source, nindent=example.indent,
                    want=example.want, got=got, test_lineno=test.lineno,
                    example_lineno=example.lineno)
        doctestplus_diffhook(info=info)
        if not info["use"]:
            return

        self._changesets[info["filename"]].append(info)
