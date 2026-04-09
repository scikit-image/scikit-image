"""Utilities for migration from ``skimage`` to ``skimage2``"""

from functools import wraps, partial
import re
from textwrap import dedent, indent


# URL to migration page.
MIGRATION_URL = (
    'https://scikit-image.org/docs/stable/user_guide/skimage2_migration.html'
)


# Identify sections specific to warning or doc.
_PARTS_RE = re.compile(
    r'''
    ^[ \t]*<!--+\ *cond-start\ *:\ *([a-z,]+)\ *--+>\ *\n
    (?P<content>.*?)\n
    [ \t]*<!--+\ *cond-end\ *--+>\ *(\n|$)
    ''',
    flags=re.DOTALL | re.MULTILINE | re.VERBOSE,
)


# Regex to find Python blocks within start-end markers.
_PYTHON_RE = re.compile(
    r'''
    ^(?P<indent>[ \t]*)
    (?P<ticks>```+)\ *
    (?P<ocurl>{)?  # Optional opening curlies
    \ *[Pp]ython\ *
    (?(ocurl)})  # Match close curlies only if opening curlies found
    \ *\n
    (?P<content>.*?)\n  # Code between start-end markers.
    (?P=indent)(?P=ticks)\ *(\n|$)  # Match indentation and backtick lengths.
    ''',
    flags=re.DOTALL | re.MULTILINE | re.VERBOSE,
)


_SKI1PREFIX_RE = re.compile(r'^skimage\.')


class Skimage2Migration:
    """Class to decorate ``skimage`` routines with migration messages

    Migration messages are in Markdown format.  You can specify which parts of
    the message become the emitted warning, and go to the migration document,
    using *conditional inclusion* start and end markers, of form ``<!---
    cond-start: warning>`` followed by text for the warning only, followed by
    ``<!--- cond-end -->``.

    Similarly, you can specify text that will only go in the migration document
    by using ``<!--- cond-start: doc>`` followed by text for the migration
    document only, followed by ``<!--- cond-end -->``.

    The text may use old-style formatting markers, with the following values
    defined:

    * ``ski1qual`` : the full qualified name of the function in the ``skimage``
      namespace.
    * ``ski2qual`` : the full qualified name of the function in the ``skimage2``
      namespace.

    Use these with old-style format specifiers such as ``%(ski1qual)s is
    deprecated in favor of ``%(ski2qual)s``.
    """

    def __init__(self, migration_url):
        self.migration_url = migration_url
        self.migration_docs = {}

    def _filled_docs(self, migration_doc, params):
        w_str, m_str = self._parse_migration_doc(migration_doc)
        return (s % params for s in (w_str, m_str))

    def _parse_migration_doc(self, doc):
        """Parse Markdown migration string to give warning and doc fragment"""
        warn_rep, doc_rep = (
            partial(self._context_rep, ctx) for ctx in ('warning', 'doc')
        )
        warn_msg = dedent(
            _PYTHON_RE.sub(self._pyblock_rep, _PARTS_RE.sub(warn_rep, doc))
        ).strip()
        if warn_msg:
            warn_msg += '\n\nSee %(migration_url)s#%(ski1qual_anchor)s'
        return warn_msg, dedent(_PARTS_RE.sub(doc_rep, doc)).strip()

    def _for_anchor(self, name):
        return name.replace('.', '-').replace('_', '-')

    def _get_func_params(self, func, ski1qual=None, ski2qual=None):
        qualname, modname = func.__qualname__, func.__module__
        ski1qual = f'{modname}.{qualname}' if ski1qual is None else ski1qual
        ski2qual = (
            _SKI1PREFIX_RE.sub(r'skimage2.', ski1qual) if ski2qual is None else ski2qual
        )
        return dict(
            qual=qualname,
            mod=modname,
            ski1qual=ski1qual,
            ski1qual_anchor=self._for_anchor(ski1qual),
            ski2qual=ski2qual,
            migration_url=self.migration_url,
        )

    def _pyblock_rep(self, match):
        return indent(match.group('content') + '\n', '  ')

    def _context_rep(self, context, match):
        doc_types = [t.strip() for t in match.group(1).split(',')]
        return '' if context not in doc_types else match.group('content') + '\n'

    def __call__(self, migration_doc, ski1qual=None, ski2qual=None):
        """Use `migration_doc` to specify warning and migration doc section

        Parameters
        ----------
        migration_doc : str
            Markdown document that may have contain conditional inclusion start
            and end markers.
        ski1qual : None or str
            The canonical full (qualified) name in the ``skimage`` namespace,
            including the ``skimage`` prefix. If None, use the functions full
            qualified name.
        ski2qual : None or str
            The matching canonical full (qualified) name in the ``skimage2``
            namespace, and including the ``skimage2`` prefix. If None, derive
            from `ski1qual`, replacing initial ``skimage.`` with ``skimage2.``.
        """

        def decorator(func):
            func_params = self._get_func_params(func, ski1qual, ski2qual)
            warn_msg, doc = self._filled_docs(migration_doc, func_params)
            if doc:
                self.migration_docs[func_params['ski1qual']] = doc

            @wraps(func)
            def decorated(*args, **kwargs):
                from skimage._shared._warnings import warn_external
                from skimage.util import PendingSkimage2Change

                warn_external(warn_msg, category=PendingSkimage2Change)
                return func(*args, **kwargs)

            return decorated

        return decorator


ski2_migration_dec = Skimage2Migration(MIGRATION_URL)
