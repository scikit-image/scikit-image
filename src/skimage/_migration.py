"""Utilities for migration from ``skimage`` to ``skimage2``"""

from functools import wraps
import re
from textwrap import dedent


# URL to migration page.
MIGRATION_URL = (
    'https://scikit-image.org/docs/stable/user_guide/skimage2_migration.html'
)

# Match blocks specific to the "warning" or "doc" context.
# These are enclosed with "context markers".
_CONTEXT_BLOCK_RE = re.compile(
    r'''
    ^[ \t]*<!--+\ *cond-start\ *:\ *(?P<context>[a-z,]+)\ *--+>\ *\n
    (?P<content>.*?)\n
    [ \t]*<!--+\ *cond-end\ *--+>\ *(\n|$)
    ''',
    flags=re.DOTALL | re.MULTILINE | re.VERBOSE,
)

# Matches any (multiline) content enclosed in double backticks.
# Content is captured in `<tickstuff>` group without backticks
_RST_LITERAL_RE = re.compile(
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


# Match restructuredText directive and its content.
_RST_DIRECTIVE_RE = re.compile(
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


_SKI1PREFIX_RE = re.compile(r'^skimage\.')


def _select_blocks(doc, *, context_name):
    """Select subset of `doc` that matches a desired context.

    Parameters
    ----------
    doc : str
        A multiline string with *context blocks*.
    context_name : str
        The name of the context block to keep. All other context blocks are
        removed.

    Returns
    -------
    context : str
        The subset of `doc` that matches the desired *context*.
    """

    def repl(match):
        doc_types = [t.strip() for t in match.group('context').split(',')]
        replace_with = ""
        if context_name in doc_types:
            replace_with = match.group('content') + '\n'
        return replace_with

    context = _CONTEXT_BLOCK_RE.sub(repl, doc)
    return context


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

        # Select blocks for "warning" context
        warn_msg = _select_blocks(doc, context_name="warning")
        # Double to single backticks
        warn_msg = _RST_LITERAL_RE.sub(r'`\g<tickstuff>`', warn_msg)
        # Clear any directives
        warn_msg = _RST_DIRECTIVE_RE.sub(r'\g<content_indent>', warn_msg)
        warn_msg = dedent(warn_msg).strip()

        if warn_msg:
            warn_msg += '\n\nSee %(migration_url)s#%(qname_old_anchor)s'

        # Select blocks for "doc" context, restore indent
        doc_rep = _select_blocks(doc, context_name="doc")
        doc_rep = dedent(doc_rep).strip()

        # Make sure that no context marker was missed
        CONTEXT_MARKER = "<!--"
        for msg, msg_type in ((warn_msg, 'warning'), (doc_rep, 'docstring')):
            if CONTEXT_MARKER in msg:
                raise ValueError(
                    f"Remaining {CONTEXT_MARKER} marker in {msg_type} "
                    f"of `{func_uri}`; check markup:\n{msg}"
                )
        return warn_msg, doc_rep

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

        if qname_old is None:
            qname_old = f'{modname}.{qualname}'
        if qname_new is None:
            qname_new = _SKI1PREFIX_RE.sub(r'skimage2.', qname_old)

        qname_old_anchor = qname_old.replace('.', '-').replace('_', '-')

        return dict(
            qual=qualname,
            mod=modname,
            qname_old=qname_old,
            qname_old_anchor=qname_old_anchor,
            qname_new=qname_new,
            migration_url=self.migration_url,
        )

    def __call__(
        self, migration_doc, *, qname_old=None, qname_new=None, warning_cls=None
    ):
        """Use `migration_doc` to specify warning and migration doc section.

        Parameters
        ----------
        migration_doc : str
            Markdown document that may have contain conditional inclusion start
            and end markers.
        qname_old : None or str, optional
            The canonical full (qualified) name in the ``skimage`` namespace,
            including the ``skimage`` prefix. If None, use the functions full
            qualified name.
        qname_new : None or str, optional
            The matching canonical full (qualified) name in the ``skimage2``
            namespace, and including the ``skimage2`` prefix. If None, derive
            from `qname_old`, replacing initial ``skimage.`` with ``skimage2.``.
        warning_cls : type[Warning], optional
            The warning class to use. Defaults to
            :obj:`~.PendingSkimage2Change` if not given.

        Returns
        -------
        decorator : Callable
            A decorator to apply to callables.
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
                    warn_external(
                        warn_msg, category=warning_cls or PendingSkimage2Change
                    )

                return func(*args, **kwargs)

            return decorated

        return decorator


ski2_migration_decorator = Skimage2Migration(MIGRATION_URL)
