"""Adapt ``_skimage2`` doctest examples for the public ``skimage`` namespace.

Implementation docstrings in ``_skimage2`` use internal imports such as ``from
_skimage2 import data``. `skimage` shims call :func:`adapt_doctests` so
``skimage`` users see and run ``skimage`` examples instead.

:func:`adapt_obj_doctest` never mutates implementation objects from
``_skimage2``. Functions get thin wrappers; classes get a local subclass proxy
whose ``__module__`` is the shim path (for pytest doctest collection).

:func:`adapt_doctests` also copies ``__doctest_requires__`` and
``__doctest_skip__`` from ``_skimage2`` implementation modules onto the shim
module so ``pytest-doctestplus`` can skip examples with missing optional
dependencies, and adds ``np`` to the shim module namespace for doctest runners.
"""

from __future__ import annotations

import functools
import inspect
import re
import types
from collections.abc import Mapping

# For namespace injection
import numpy as np  # noqa: F401


__all__ = ['adapt_doctest_doc', 'adapt_obj_doctest', 'adapt_doctests']

_DOCTEST_PROMPT_RE = re.compile(
    r'''
    ^(?P<indent>\s*)
    (>>> )
    ''',
    flags=re.VERBOSE,
)
# Seearch / replace pairs.
_SEARCH_REP_PAIRS = (
    (re.compile(r'import _?skimage2 as ski2'), 'import skimage as ski'),
    (re.compile(r'ski2\.'), 'ski.'),
    (
        re.compile(
            r'''
                (?P<fi>from|import)\ +
                (_?skimage2)
                (?P<connect>[ .])''',
            flags=re.VERBOSE,
        ),
        r'\g<fi> skimage\g<connect>',
    ),
    (re.compile(r'_?skimage2\.'), 'skimage.'),
)


def adapt_doctest_doc(doc: str | None) -> str | None:
    """Rewrite ``_skimage2`` / ``skimage2`` inside doctest blocks only."""
    if not doc:
        return doc

    lines: list[str] = doc.split('\n')
    out: list[str] = []

    while lines:
        line = lines.pop(0)
        if not (match := _DOCTEST_PROMPT_RE.match(line)):
            out.append(line)
            continue
        n_block_indent: int = len(match.group('indent'))
        out.append(_proc_line(line))
        in_doctest_block: bool = True

        while in_doctest_block and lines:
            line = lines.pop(0)
            n_indent = len(line) - len(line.lstrip())
            in_doctest_block = line and (n_indent >= n_block_indent)
            out.append(_proc_line(line) if in_doctest_block else line)

    return '\n'.join(out)


def _proc_line(line):
    for regx, subs in _SEARCH_REP_PAIRS:
        line = regx.sub(subs, line)
    return line


def adapt_obj_doctest(obj, shim_module: str = None):
    """Return a shim-local view with adapted doctest examples."""
    if not (inspect.isroutine(obj) or inspect.isclass(obj)):
        return obj
    original_doc = getattr(obj, '__doc__', None)
    adapted_doc = adapt_doctest_doc(original_doc)
    if adapted_doc == original_doc:
        return obj
    return _with_adapted_doc(obj, adapted_doc, shim_module)


def adapt_doctests(
    ns: Mapping[str, object],
    *,
    skip_names: tuple[str, ...] = (),
) -> None:
    """Adapt docstrings for callables and classes exported from a shim module."""
    if not isinstance(ns, dict):
        raise TypeError('adapt_doctests expects a mutable mapping such as globals()')
    shim_module = ns.get('__name__')
    skip = set(skip_names)
    # Collect implementing modules for functions / classes in namespace.
    # Adapt doctests for functions / classes.
    impl_modules: set[types.ModuleType] = set()
    for name, obj in list(ns.items()):
        if name.startswith('__') or name in skip or inspect.ismodule(obj):
            continue
        if not (inspect.isroutine(obj) or inspect.isclass(obj)):
            continue
        impl_module = inspect.getmodule(obj)
        if impl_module is not None:
            impl_modules.add(impl_module)
        ns[name] = adapt_obj_doctest(obj, shim_module=shim_module)

    # Apply doctest-plus markers from original implementions.
    if impl_modules:
        _sync_doctest_markers(ns, impl_modules, shim_module)

    module_doc = ns.get('__doc__')
    adapted_module_doc = adapt_doctest_doc(module_doc)
    if adapted_module_doc != module_doc:
        ns['__doc__'] = adapted_module_doc

    # Add NumPy `np` to module namespace for doctest compatibility.
    ns.setdefault('np', np)


def _sync_doctest_markers(
    ns: dict,
    impl_modules: set[types.ModuleType],
    shim_module: str,
) -> None:
    """Copy pytest-doctestplus skip metadata from implementation modules."""
    requires = dict(ns.get('__doctest_requires__', {}))
    skip = list(ns.get('__doctest_skip__', []))

    for impl_module in impl_modules:
        for key, mods in getattr(impl_module, '__doctest_requires__', {}).items():
            merged = requires.setdefault(key, [])
            merged.extend(m for m in mods if m not in merged)

        for pattern in getattr(impl_module, '__doctest_skip__', []):
            if pattern not in skip:
                skip.append(pattern)

    if requires:
        ns['__doctest_requires__'] = requires
    if skip:
        ns['__doctest_skip__'] = skip


def _with_adapted_doc(obj, adapted_doc: str, shim_module: str):
    if inspect.isroutine(obj):
        if isinstance(obj, types.FunctionType):
            new_func = types.FunctionType(
                obj.__code__,
                obj.__globals__,
                obj.__name__,
                obj.__defaults__,
                obj.__closure__,
            )
            functools.update_wrapper(new_func, obj)
            new_func.__doc__ = adapted_doc
            new_func.__module__ = shim_module
            return new_func

        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            return obj(*args, **kwargs)

        wrapper.__doc__ = adapted_doc
        wrapper.__module__ = shim_module
        return wrapper

    proxy = types.new_class(
        obj.__name__,
        (obj,),
        {},
        lambda ns: ns.update({'__doc__': adapted_doc}),
    )
    proxy.__module__ = shim_module
    proxy.__qualname__ = obj.__qualname__
    return proxy
