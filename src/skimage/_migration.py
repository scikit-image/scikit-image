"""Utilities for migration from ``skimage`` to ``skimage2``"""

from functools import wraps
import re
from textwrap import dedent


# URL to migration page.
MIGRATION_URL = (
    'https://scikit-image.org/docs/stable/user_guide/skimage2_migration.html'
)

COMMENT_MARKER = "<!--"

# Identify sections specific to warning or doc.
_PARTS_RE = re.compile(
    r'''
    ^[ \t]*<!--+\ *cond-start\ *:\ *(?P<doctypes>[a-z,]+)\ *--+>\ *\n
    (?P<content>.*?)\n
    [ \t]*<!--+\ *cond-end\ *--+>\ *(\n|$)
    ''',
    flags=re.DOTALL | re.MULTILINE | re.VERBOSE,
)

_TWO_BT_RE = re.compile(
    r'''
    (?<!`) # Not `
    ``     # Two ``
    (?!`)  # Not `
    (?P<tickstuff>.*?)
    (?<!`) # Not `
    ``     # Two ``
    (?!`)  # Not `
    ''',
    flags=re.DOTALL | re.MULTILINE | re.VERBOSE,
)

_DIRECTIVE_RE = re.compile(
    r'''
    ^(?P<indent>[ \t]*)   # Indent
    \.\.[ \t]+            # Up to directive name.
    [a-zA-Z0-9_\-]+       # Directive name.
    [ \t]*::              # up to :: delimiter.
    [ \t].*?\n            # To end of line, including on-line params.
    ((?P=indent)[ \t]+:   # Up to parameter name
    [a-zA-Z0-9_\-]+       # Parameter name.
    :.*?\n)*              # None or more parameters.
    \s*                   # None or more spaces, including newlines.
    (?P<content_indent>   # Group for content indent.
    ^(?P=indent)[ \t]+    # Up to indent and more whitespace.
    )
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

# Regex to find Python code-blocks within code-block markers.
# This allows us to identify code that should be executed.
_PYTHON_CB_RE = re.compile(
    r'''
    ^(?P<indent>[ \t]*)
    (?P<ticks>```+)\ *
    {\ *code-block\ *}
    \ *[Pp]ython\ *
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

    * ``qname_old`` : the full qualified name of the function in the ``skimage``
      namespace.
    * ``qname_new`` : the full qualified name of the function in the ``skimage2``
      namespace.

    Use these with old-style format specifiers such as ``%(qname_old)s is
    deprecated in favor of ``%(qname_new)s``.
    """

    def __init__(self, migration_url):
        self.migration_url = migration_url
        self.migration_docs = {}
        self.extra_params = {}  # For useful extra parameters in migration doc.
        self.doctests = {}

    def _filled_docs(self, migration_doc, params):
        """Parse and do substitutions on input `migration_doc`

        Parameters
        ----------
        migration_doc : str
            Input migration doc.
        params : dict
            Dictionary of parameters to be used for substitution into the
            parsed warning and docstring.

        Returns
        -------
        warn_str : str
            Parsed warning string, with any substitutions.
        doct_str : str
            Parsed docstring, with any substitutions.
        """
        w_str, m_str = self._parse_migration_doc(migration_doc, params['qname_old'])
        return (s % params for s in (w_str, m_str))

    def _parse_migration_doc(self, doc, func_uri=None):
        """Parse Markdown migration string to give warning and doc fragment"""
        warn_rep, doc_rep = (self._context_rep(ctx) for ctx in ('warning', 'doc'))
        warn_msg = dedent(
            _DIRECTIVE_RE.sub(
                r'\g<content_indent>',  # Clear any directives, restore indent.
                _TWO_BT_RE.sub(
                    r'`\g<tickstuff>`',  # Double to single backticks.
                    _PARTS_RE.sub(
                        warn_rep,  # Remove not-matching context.
                        doc,
                    ),
                ),
            )
        ).strip()
        if warn_msg:
            warn_msg += '\n\nSee %(migration_url)s#%(qname_old_anchor)s'
        doc_rep = dedent(_PARTS_RE.sub(doc_rep, doc)).strip()
        for msg, msg_type in ((warn_msg, 'warning'), (doc_rep, 'docstring')):
            if COMMENT_MARKER in msg:
                raise ValueError(
                    f"Remaining {COMMENT_MARKER} marker in {msg_type} "
                    f"of `{func_uri}`; check markup:\n{msg}"
                )
        return warn_msg, doc_rep

    def _for_anchor(self, name):
        """Rewrite function URI `name` to be valid as ReST link"""
        return name.replace('.', '-').replace('_', '-')

    def _get_func_params(self, func, qname_old=None, qname_new=None):
        """Compile dictionary of parameters from input `func`

        Parameters
        ----------
        func : function
        qname_old : None or str
            If specified, use as the input original (pre-migration) name.
        qname_new : None or str
            If specified, use as the input migrated (post-migration) name.

        Returns
        -------
        params : dict
            Dictionary of parameters.
        """
        qualname, modname = func.__qualname__, func.__module__
        qname_old = f'{modname}.{qualname}' if qname_old is None else qname_old
        qname_new = (
            _SKI1PREFIX_RE.sub(r'skimage2.', qname_old)
            if qname_new is None
            else qname_new
        )
        return dict(
            qual=qualname,
            mod=modname,
            qname_old=qname_old,
            qname_old_anchor=self._for_anchor(qname_old),
            qname_new=qname_new,
            migration_url=self.migration_url,
        )

    def _context_rep(self, context):
        """Make replacer function to select blocks based on `context`.

        Returns replacer function for use with e.g ``re.sub``.

        Replacer functions return strings given an input :class:`re.Match`
        instance.

        In this case, each ``match`` corresponds to what we will call a
        "block".

        `context` is a string that specifies which types of content blocks the
        returned replacer function should select.

        The returned replacer function assumes the ``match`` has a group
        "doctypes" that is a comma-separated set of "contexts" corresponding to
        this "block".  The replacer discards the block (returns ``''``) for any
        blocks where `context` is not in the "doctypes" set of contexts, and
        returns the "content" group of the match otherwise.

        Parameters
        ----------
        context : str
            `context` specifies the replacement context.

        Returns
        -------
        replacer_func : function
            Replacer function for use with ``re.sub`` etc.
        """

        def rep_func(match):
            doc_types = [t.strip() for t in match.group('doctypes').split(',')]
            return '' if context not in doc_types else match.group('content') + '\n'

        return rep_func

    def __call__(self, migration_doc, qname_old=None, qname_new=None):
        """Use `migration_doc` to specify warning and migration doc section

        Parameters
        ----------
        migration_doc : str
            Markdown document that may have contain conditional inclusion start
            and end markers.
        qname_old : None or str
            The canonical full (qualified) name in the ``skimage`` namespace,
            including the ``skimage`` prefix. If None, use the functions full
            qualified name.
        qname_new : None or str
            The matching canonical full (qualified) name in the ``skimage2``
            namespace, and including the ``skimage2`` prefix. If None, derive
            from `qname_old`, replacing initial ``skimage.`` with ``skimage2.``.
        """

        def decorator(func):
            """Decorate `func`"""
            func_params = self._get_func_params(func, qname_old, qname_new)
            warn_msg, doc = self._filled_docs(migration_doc, func_params)
            if doc:
                self.migration_docs[func_params['qname_old']] = doc

            @wraps(func)
            def decorated(*args, **kwargs):
                from _skimage2._shared._warnings import warn_external
                from skimage.util import PendingSkimage2Change

                if warn_msg:
                    warn_external(warn_msg, category=PendingSkimage2Change)

                return func(*args, **kwargs)

            return decorated

        return decorator


ski2_migration_decorator = Skimage2Migration(MIGRATION_URL)
