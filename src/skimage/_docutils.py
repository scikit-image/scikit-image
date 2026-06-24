"""Adapt ``_skimage2`` doctest examples for the public ``skimage`` namespace.

Implementation docstrings in ``_skimage2`` use internal imports such as
``from _skimage2 import data``. Shims call :func:`bind_public` or
:func:`bind_namespace` so ``skimage`` users see and run ``skimage`` examples
instead.

To leave a doctest block unchanged, put the literal token
``#: skimage-shim-keep-doctest`` in a comment on the first ``>>>`` line of
that block. This is not a doctest directive; :func:`adapt_doctest_doc` searches
for that exact string only.

:func:`bind_public` never mutates implementation objects from ``_skimage2``.
Functions get thin wrappers; classes get a local subclass proxy whose
``__module__`` is the shim path (for pytest doctest collection). Implementation
``__module__`` and ``__doc__`` stay on the canonical ``_skimage2`` object so a
future pickle compatibility layer can own stable public metadata separately.
"""

from __future__ import annotations

import functools
import inspect
import re
import types
from collections.abc import Mapping

__all__ = ['adapt_doctest_doc', 'bind_public', 'bind_namespace']

_KEEP_DOCTEST_MARKER = '#: skimage-shim-keep-doctest'
_DOCTEST_PROMPT_RE = re.compile(r'^(\s*)(>>>|\.\.\.)')
_SKIMAGE2_RE = re.compile(r'_skimage2')
_SKIMAGE2_PKG_RE = re.compile(r'\bskimage2\b')


def adapt_doctest_doc(doc: str | None, *, skip: bool = False) -> str | None:
    """Rewrite ``_skimage2`` / ``skimage2`` inside doctest blocks only.

    Put ``#: skimage-shim-keep-doctest`` in a comment on the first ``>>>`` line
    of a block to leave that whole block unchanged. This token is not a doctest
    directive; it is matched as a literal string only.
    """
    if doc is None or skip:
        return doc

    lines = doc.split('\n')
    out: list[str] = []
    index = 0

    while index < len(lines):
        match = _DOCTEST_PROMPT_RE.match(lines[index])
        if match is None:
            out.append(lines[index])
            index += 1
            continue

        block_indent = len(match.group(1))
        keep = _KEEP_DOCTEST_MARKER in lines[index]
        block = [lines[index]]
        index += 1
        while index < len(lines) and _line_in_doctest_block(block_indent, lines[index]):
            block.append(lines[index])
            index += 1

        if keep:
            out.extend(block)
        else:
            out.extend(map(_rewrite_doctest_line, block))

    return '\n'.join(out)


def bind_public(obj, *, shim_module: str | None = None):
    """Return a shim-local view with adapted doctest examples.

    ``_skimage2`` implementations are never modified. Routines get thin
    wrappers; classes get local subclass proxies with ``__module__`` set to the
    shim path for pytest doctest collection.
    """
    if obj is None:
        return obj

    original_doc = getattr(obj, '__doc__', None)
    adapted_doc = adapt_doctest_doc(original_doc)
    if adapted_doc == original_doc:
        return obj

    if inspect.isroutine(obj) or inspect.isclass(obj):
        return _with_public_doc(obj, adapted_doc, shim_module)
    return obj


def bind_namespace(
    ns: Mapping[str, object],
    *,
    skip_names: tuple[str, ...] = (),
) -> None:
    """Adapt docstrings for callables and classes exported from a shim module."""
    if not isinstance(ns, dict):
        raise TypeError('bind_namespace expects a mutable mapping such as globals()')

    shim_module = ns.get('__name__')
    if not isinstance(shim_module, str):
        shim_module = None

    skip = set(skip_names)
    for name, obj in list(ns.items()):
        if name.startswith('__') or name in skip or inspect.ismodule(obj):
            continue
        if inspect.isroutine(obj) or inspect.isclass(obj):
            ns[name] = bind_public(obj, shim_module=shim_module)

    module_doc = ns.get('__doc__')
    adapted_module_doc = adapt_doctest_doc(module_doc)
    if adapted_module_doc != module_doc:
        ns['__doc__'] = adapted_module_doc


def _rewrite_doctest_line(line: str) -> str:
    line = _SKIMAGE2_RE.sub('skimage', line)
    return _SKIMAGE2_PKG_RE.sub('skimage', line)


def _line_in_doctest_block(block_indent: int, line: str) -> bool:
    if not line.strip():
        return False
    if _DOCTEST_PROMPT_RE.match(line):
        return True
    return len(line) - len(line.lstrip()) >= block_indent


def _with_public_doc(obj, adapted_doc: str | None, shim_module: str | None):
    if inspect.isroutine(obj):

        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            return obj(*args, **kwargs)

        wrapper.__doc__ = adapted_doc
        if shim_module is not None:
            wrapper.__module__ = shim_module
        return wrapper

    proxy = types.new_class(
        obj.__name__,
        (obj,),
        {},
        lambda ns: ns.update({'__doc__': adapted_doc}),
    )
    proxy.__module__ = shim_module or obj.__module__
    proxy.__qualname__ = obj.__qualname__
    return proxy
