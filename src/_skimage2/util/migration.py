"""Utilities for migration from ``skimage`` to ``skimage2``"""

from functools import wraps, partial
import re
from textwrap import dedent

_PARTS_RE = re.compile(r'''\
<!--+\s*cond-start\s*:\s*([a-z,]+)\s*--+>\s*$
(.*?)\s*\n
<!--+\s*cond-end\s*--+>\s*$''',
                       flags=re.DOTALL | re.MULTILINE | re.VERBOSE)

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

    def __init__(self, warn=True):
        self.warn = warn
        self.migration_docs = {}

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
            filled = dedent(migration_doc % func_params)
            warn_msg, doc = self._parse_migration_doc(filled)
            self.migration_docs[func_params['ski1qual']] = doc

            @wraps(func)
            def decorated(*args, **kwargs):
                from skimage._shared._warnings import warn_external
                from skimage.util import PendingSkimage2Change

                if self.warn:
                    warn_external(warn_msg, category=PendingSkimage2Change)
                return func(*args, **kwargs)

            return decorated

        return decorator

    def _get_func_params(self, func, ski1qual=None, ski2qual=None):
        qualname, modname = func.__qualname__, func.__module__
        ski1qual = f'{modname}.{qualname}' if ski1qual is None else ski1qual
        ski2qual = (
            _SKI1PREFIX_RE.sub(r'skimage2.', ski1qual) if ski2qual is None
            else ski2qual
        )
        return dict(qual=qualname, mod=modname, ski1qual=ski1qual, ski2qual=ski2qual)

    def _replacer(self, context, match):
        doc_types = [t.strip() for t in match.group(1).split(',')]
        return '' if context not in doc_types else match.group(2)

    def _parse_migration_doc(self, doc):
        """Parse Markdown migration string to give warning and doc fragment
        """
        return (_PARTS_RE.sub(partial(self._replacer, 'warning'), doc),
                _PARTS_RE.sub(partial(self._replacer, 'doc'), doc))


# Change warn=True when skimage2 namespace is ready.
ski2_migration_dec = Skimage2Migration(warn=False)
