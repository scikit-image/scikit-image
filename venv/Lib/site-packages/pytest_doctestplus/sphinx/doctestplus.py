# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This is a set of three directives that allow us to insert metadata
about doctests into the .rst files so the testing framework knows
which tests to skip.

This is quite different from the doctest extension in Sphinx itself,
which actually does something.  For astropy, all of the testing is
centrally managed from py.test and Sphinx is not used for running
tests.
"""
import re
from docutils.parsers.rst import Directive
from sphinx.util.docutils import SphinxDirective

class NoRunDirective(Directive):
    def run(self):
        # Simply do not add any content when this directive is encountered
        return []


class DoctestSkipDirective(SphinxDirective):
    has_content = True

    def run(self):
        # Check if there is any valid argument, and skip it. Currently only
        # 'win32' is supported.
        if len(self.content) > 0 and re.match("win32", self.content[0]):
            self.content = self.content[2:]

        nodes = self.parse_content_to_nodes()
        return nodes

class DoctestOmitDirective(NoRunDirective):
    has_content = True


class DoctestRequiresDirective(DoctestSkipDirective):
    # This is silly, but we really support an unbounded number of
    # optional arguments
    optional_arguments = 64


class DoctestAllDirective(NoRunDirective):
    optional_arguments = 64
    has_content = False


def setup(app):

    app.add_directive('doctest-requires', DoctestRequiresDirective)
    app.add_directive('doctest-requires-all', DoctestAllDirective)
    app.add_directive('doctest-skip', DoctestSkipDirective)
    app.add_directive('doctest-skip-all', DoctestAllDirective)
    app.add_directive('doctest', DoctestSkipDirective, override=True)
    app.add_directive('doctest-remote-data', DoctestSkipDirective)
    app.add_directive('doctest-remote-data-all', DoctestAllDirective)
    # Code blocks that use this directive will not appear in the generated
    # documentation. This is intended to hide boilerplate code that is only
    # useful for testing documentation using doctest, but does not actually
    # belong in the documentation itself.
    app.add_directive('testsetup', DoctestOmitDirective, override=True)
    app.add_directive('testcleanup', DoctestOmitDirective, override=True)

    return {'parallel_read_safe': True,
            'parallel_write_safe': True}
