"""Utilities for migration from ``skimage`` to ``skimage2``"""

import re
from functools import wraps

_HEADINGS = ('Summary', 'Examples', 'Background')

_HEADINGS_RE = re.compile(
    rf'''
    ^
    \#+\s+({'|'.join(_HEADINGS)})
    \s*
    $
    ''',
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)

_SKI1PREFIX_RE = re.compile(r'^skimage\.')


class Skimage2Migration:
    """Class to decorate ``skimage`` routines with migration messages"""

    def __init__(self, warn=True):
        self.warn = warn
        self.migration_messages = {}

    def __call__(self, migration_doc, ski2qual=None):
        """Use `migration_doc` to specify warning and migration doc section"""

        def decorator(func):
            func_params = self._get_func_params(func, ski2qual)
            filled = migration_doc % func_params
            parts = self._parse_migration_doc(filled)
            self.migration_messages[func_params['ski1qual']] = parts

            @wraps(func)
            def decorated(*args, **kwargs):
                from skimage._shared._warnings import warn_external
                from skimage.util import PendingSkimage2Change

                if self.warn:
                    warn_external(parts['Summary'], category=PendingSkimage2Change)
                return func(*args, **kwargs)

            return decorated

        return decorator

    def _get_func_params(self, func, ski2qual=None):
        qualname, modname = func.__qualname__, func.__module__
        ski1qual = f'{modname}.{qualname}'
        ski2qual = (
            _SKI1PREFIX_RE.sub(r'skimage2.', ski1qual) if ski2qual is None else ski2qual
        )
        return dict(qual=qualname, mod=modname, ski1qual=ski1qual, ski2qual=ski2qual)

    def _parse_migration_doc(self, doc):
        """Parse Markdown migration string `doc` into parts"""
        head_split = _HEADINGS_RE.split(doc)[1:]
        parts = dict(
            zip(
                (p.capitalize() for p in head_split[::2]),
                (p.strip() for p in head_split[1::2]),
            )
        )
        if 'Summary' not in parts:
            raise ValueError('Migration message should contain a summary')
        return parts


# Change warn=True when skimage2 namespace is ready.
ski2_migration_dec = Skimage2Migration(warn=False)
