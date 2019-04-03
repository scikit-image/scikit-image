"""
This file was adapted from scipy's file with the same name. Since it was only
slightly modified, their copyright notice is included at the bottom of the file.


From their header:
Tests which scan for certain occurrences in the code, they may not find
all of these occurrences but should catch almost all. This file was adapted
from numpy.

"""


from __future__ import division, absolute_import, print_function

import os
import sys
import skimage

import pytest


from pathlib import Path
import ast
import tokenize

file_whitelist = [
    "_shared/tests/test_warnings.py", # this file
    "conftest.py",
]

class ParseCall(ast.NodeVisitor):
    def __init__(self):
        self.ls = []

    def visit_Attribute(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        self.ls.append(node.attr)

    def visit_Name(self, node):
        self.ls.append(node.id)

class FindFuncs(ast.NodeVisitor):
    def __init__(self, filename):
        super().__init__()
        self._filename = filename
        self.bad_filters = []
        self.bad_stacklevels = []

    def visit_Call(self, node):
        p = ParseCall()
        p.visit(node.func)
        ast.NodeVisitor.generic_visit(self, node)

        if str(self._filename) in file_whitelist:
            return
        if p.ls[-1] == 'simplefilter' or p.ls[-1] == 'filterwarnings':
            if node.args[0].s == "ignore":
                self.bad_filters.append(
                    "{}:{}".format(self._filename, node.lineno))

        if p.ls[-1] == 'warn' and (
                len(p.ls) == 1 or p.ls[-2] == 'warnings'):

            # See if stacklevel exists:
            if len(node.args) == 3:
                return
            args = {kw.arg for kw in node.keywords}
            if "stacklevel" not in args:
                self.bad_stacklevels.append(
                    "{}:{}".format(self._filename, node.lineno))


@pytest.fixture(scope="session")
def warning_calls():
    # combined "ignore" and stacklevel error
    base = Path(skimage.__file__).parent

    bad_filters = []
    bad_stacklevels = []

    for path in base.rglob("*.py"):
        # use tokenize to auto-detect encoding on systems where no
        # default encoding is defined (e.g. LANG='C')
        with tokenize.open(str(path)) as file:
            tree = ast.parse(file.read(), filename=str(path))
            finder = FindFuncs(path.relative_to(base))
            finder.visit(tree)
            bad_filters.extend(finder.bad_filters)
            bad_stacklevels.extend(finder.bad_stacklevels)

    return bad_filters, bad_stacklevels


# @pytest.mark.slow
def test_warning_calls_filters(warning_calls):
    bad_filters, bad_stacklevels = warning_calls

    if bad_filters:
        raise AssertionError(
            "warning ignore filter should not be in tests, instead, use\n"
            "skimage._warnings.expected_warnings;\n"
            "found in:\n    {}".format(
                "\n    ".join(bad_filters)))


# @pytest.mark.slow
def test_warning_calls_stacklevels(warning_calls):
    bad_filters, bad_stacklevels = warning_calls

    msg = ""

    if bad_filters:
        msg += ("warning ignore filter should not be used, instead, use\n"
                "scipy._lib._numpy_compat.suppress_warnings (in tests only);\n"
                "found in:\n    {}".format("\n    ".join(bad_filters)))
        msg += "\n\n"

    # tifffile still doesn't use stacklevels everywhere, but we plan to stop
    # versioning it soon anyway.
    bad_stacklevels = [item
                       for item in bad_stacklevels
                       if 'tifffile.py' not in item]

    if bad_stacklevels:
        msg += ("warnings should have an appropriate stacklevel:\n"
                "    {}".format("\n    ".join(bad_stacklevels)))

    if msg:
        raise AssertionError(msg)



"""
Copyright (c) 2001, 2002 Enthought, Inc.
All rights reserved.

Copyright (c) 2003-2017 SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of Enthought nor the names of the SciPy Developers
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.

"""
