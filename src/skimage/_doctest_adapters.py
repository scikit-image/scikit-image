"""Adapt ``_skimage2`` doctest examples for the public ``skimage`` namespace.

Implementation docstrings in ``_skimage2`` use internal imports such as ``from
_skimage2 import data``. `skimage` shims call :func:`adapt_doctests` so
``skimage`` users see and run ``skimage`` docstring examples instead.

:func:`adapt_doctests` never modifies implementation objects from
``_skimage2``.  It returns copies of functions when their doctests need
rewriting.  For classes, it always returns a subclass with the shim
``__module__``, so shim modules can safely patch the class without mutating the
implementation.  We rewrite docstrings on the subclass when they contain
``_skimage2`` / ``skimage2`` doctest references.  Each subclass also inherits
from the shim subclasses of its implementation parents, so that the shim
classes have the same hierarchy as the implementation classes.

:func:`adapt_doctests` also copies ``__doctest_requires__`` and
``__doctest_skip__`` from ``_skimage2`` implementation modules onto the shim
module so ``pytest-doctestplus`` can skip examples with missing optional
dependencies, and adds ``np`` to the shim module namespace for doctest runners.
"""

from __future__ import annotations

import functools
import inspect
import re
import sys
import types
import typing
from collections.abc import Mapping

# For namespace injection
import numpy as np  # noqa: F401


__all__ = ['adapt_doctests']

_DOCTEST_PROMPT_RE = re.compile(
    r'''
    ^(?P<indent>\s*)
    (>>> )
    ''',
    flags=re.VERBOSE,
)

# Search / replace pairs.
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


def _adapt_doctest_doc(doc: str | None) -> str | None:
    """Rewrite ``_skimage2`` / ``skimage2`` inside doctest blocks only.

    Parameters
    ----------
    doc : str
        Docstring with doctests changed to use `skimage` instead of
        ``_skimage2`` / ``_skimage2``.

    Returns
    -------
    new_doc : str
        Docstring modified to change ``_skimage2`` / ``skimage2`` examples in
        doctests to use ``skimage`` instead.  If `doc` is None or empty, ``doc
        is new_doc``, otherwise `new_doc` is a new string.
    """
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


# Shim class for each implementation class.
_IMPL_TO_SHIM: dict[type, type] = {}


def _shim_class(impl_class, mod_name):
    """Return the shim class for `impl_class`, making it if necessary.

    The shim classes for the ``_skimage2`` bases of `impl_class` must already
    exist, so that the shim classes have the same hierarchy as the
    implementation classes.
    """
    if impl_class in _IMPL_TO_SHIM:
        return _IMPL_TO_SHIM[impl_class]
    shim_parents = []
    for base in impl_class.__bases__:
        if not _is_from_skimage2(base):
            continue
        if base not in _IMPL_TO_SHIM:
            raise TypeError(
                f'No shim class for {base.__module__}.{base.__qualname__}, '
                f'base of {impl_class.__qualname__}; adapt the shim module for '
                f'{base.__qualname__} before {mod_name!r}'
            )
        shim_parents.append(_IMPL_TO_SHIM[base])
    class_doc = _adapt_doctest_doc(impl_class.__doc__)
    return _adapt_class(impl_class, shim_parents, class_doc, mod_name)


def _make_inheritor(impl_class, shim_parents, doc, mod_name):
    """Return shim class that inherits from `impl_class`

    The shim class also inherits from `shim_parents`.  If `impl_class` is a
    protocol, the shim class is a protocol too.

    Parameters
    ----------
    impl_class: type
        Implementation class.
    shim_parents: sequence of type
        Shim classes of the bases of `impl_class`.
    doc: str
        Class docstring for shim class.
    mod_name: str
        Module name for shim class.

    Returns
    -------
    shim_class: type
        A class that inherits from `impl_class`, registered in
        ``_IMPL_TO_SHIM``.
    """
    bases = (impl_class, *shim_parents)
    if getattr(impl_class, '_is_protocol', False):
        bases += (typing.Protocol,)
    shim_class = types.new_class(impl_class.__name__, bases, {})
    shim_class.__doc__ = doc
    shim_class.__module__ = mod_name
    shim_class.__qualname__ = impl_class.__qualname__
    if getattr(impl_class, '_is_runtime_protocol', False):
        shim_class = typing.runtime_checkable(shim_class)
    _IMPL_TO_SHIM[impl_class] = shim_class
    return shim_class


def _is_from_shim(obj, shim_module: str):
    """True if `obj` is owned by `shim_module`."""
    return getattr(obj, '__module__', None) == shim_module


def _is_from_skimage2(obj):
    """True if `obj` is owned by a module in `_skimage2`."""
    return obj.__module__.split('.')[0] == '_skimage2'


def shim_wraps(wrapped):
    """Like :func:`functools.wraps`, but keeps the enclosing module for ``__module__``.

    Use for shim wrappers around ``_skimage2`` functions: the wrapper inherits
    the wrapped function's name, docstring, annotations and ``__wrapped__``
    chain, but advertises itself as living in the shim module where it is
    defined.
    """

    def decorator(wrapper):
        functools.update_wrapper(wrapper, wrapped)
        wrapper.__module__ = wrapper.__globals__['__name__']
        return wrapper

    return decorator


@functools.cache
def _check_no_instance_state(impl_class, shim_class):
    """Raise TypeError if `shim_class` builds instances unlike `impl_class`.

    Python refuses ``__class__`` changes between incompatible layouts; this
    also refuses shim ``__init__`` or ``__new__`` with code of their own.
    Doctest copies of these methods share the code of the original.
    """
    for name in ('__init__', '__new__'):
        shim_code = getattr(getattr(shim_class, name), '__code__', None)
        if shim_code is not getattr(getattr(impl_class, name), '__code__', None):
            raise TypeError(f'Shim class {shim_class} must not define {name}')


def as_shim(obj):
    """Return `obj`, or each item of list `obj`, as an instance of its shim class.

    Objects of classes without a shim class pass through unchanged.  Changing
    ``__class__`` is safe because shim classes add no instance state.
    """
    if isinstance(obj, list):
        return [as_shim(item) for item in obj]
    impl_class = type(obj)
    shim_class = _IMPL_TO_SHIM.get(impl_class)
    if shim_class is not None:
        _check_no_instance_state(impl_class, shim_class)
        obj.__class__ = shim_class
    return obj


def returns_shim(func):
    """Wrap ``_skimage2`` function `func` so that it returns shim-class objects.

    The wrapper has the ``__module__`` of the calling shim module.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return as_shim(func(*args, **kwargs))

    wrapper.__module__ = sys._getframe(1).f_globals['__name__']
    return wrapper


def _adapt_obj_doctest(obj, shim_module: str):
    """Return a shim-local view with adapted doctest examples, as necessary

    If a function needs a modified docstring, we return a copy of that
    function, with the modified docstring.

    If `obj` is a class from ``_skimage2``, we return its shim class, a
    subclass, so any later shim-only patches do not mutate the implementation.
    Class and method docstrings are rewritten when they contain ``_skimage2``
    / ``skimage2`` doctest references.  All shim modules share one shim class
    for each implementation class.

    If no docstring modifications are necessary for a function, or `obj` is
    neither a function nor a class, we return the original object.  We do not
    change other classes, such as those defined in `skimage`, or in other
    packages.
    """
    # Do not adapt anything that is defined in this module.
    if _is_from_shim(obj, shim_module):
        return obj
    if inspect.isroutine(obj):
        original_doc = getattr(obj, '__doc__', None)
        adapted_doc = _adapt_doctest_doc(original_doc)
        if original_doc == adapted_doc:
            return obj
        return _copy_callable(obj, adapted_doc, shim_module)
    if not inspect.isclass(obj):
        return obj
    if not _is_from_skimage2(obj):
        # A shim class from another shim module, a class defined in skimage,
        # or a class from another package, such as a NumPy scalar type.
        return obj
    return _shim_class(obj, shim_module)


def adapt_doctests(
    ns: Mapping[str, object],
    *,
    skip_names: tuple[str, ...] = (),
) -> None:
    """Adapt docstrings for callables and classes exported from a shim module

    Implementation classes imported into `ns` are replaced with their shim
    class.  All shim modules share one shim class for each implementation
    class, owned by the first shim module that adapts it.  That protects the
    implementation from later patches in the shims, because changes are on
    the shim class.  Classes not from ``_skimage2`` are not changed.

    ns : Mapping
        Namespace of shim module.  Usually generated by passing result of
        ``globals()`` from within shim module ``__init__``.  `ns` should define
        at least the ``__name__`` element.  We modify `ns` in-place.
    skip_names : tuple of str, optional, keyword-only
        Names of objects to pass through without modification.

    Returns
    -------
    None
    """
    if not isinstance(ns, dict):
        raise TypeError('adapt_doctests expects a mutable mapping such as globals()')
    shim_module = ns.get('__name__')
    skip = set(skip_names)
    # Collect implementing modules for functions / classes in namespace.
    # Adapt doctests for functions / classes.
    impl_modules: set[types.ModuleType] = set()
    # Adapt classes before their subclasses, so that a class in this namespace
    # gets its shim here, not as the parent of another class.
    for name, obj in sorted(ns.items(), key=_mro_length):
        if name.startswith('__') or name in skip or inspect.ismodule(obj):
            continue
        if not (inspect.isroutine(obj) or inspect.isclass(obj)):
            continue
        impl_module = inspect.getmodule(obj)
        if impl_module is not None:
            impl_modules.add(impl_module)
        ns[name] = _adapt_obj_doctest(obj, shim_module=shim_module)

    # Apply doctest-plus markers from original implementions.
    if impl_modules:
        _sync_doctest_markers(ns, impl_modules, shim_module)

    module_doc = ns.get('__doc__')
    adapted_module_doc = _adapt_doctest_doc(module_doc)
    if adapted_module_doc != module_doc:
        ns['__doc__'] = adapted_module_doc

    # Add NumPy `np` to module namespace for doctest compatibility.
    ns.setdefault('np', np)


def _mro_length(item):
    """Sort key for namespace `item`: MRO length for classes, else 0."""
    name, obj = item
    if inspect.isclass(obj):
        return len(obj.__mro__)
    return 0


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


def _copy_callable(obj, adapted_doc, module_name):
    """Make copy of function `obj` with given ``__doc__`` and ``__module__``."""
    new_func = types.FunctionType(
        obj.__code__,
        obj.__globals__,
        obj.__name__,
        obj.__defaults__,
        obj.__closure__,
    )
    functools.update_wrapper(new_func, obj)
    new_func.__kwdefaults__ = obj.__kwdefaults__
    new_func.__doc__ = adapted_doc
    new_func.__module__ = module_name
    return new_func


def _adapt_class(impl_class, shim_parents, class_doc, mod_name):
    """Return a shim subclass of `impl_class` with adapted docstrings

    Always returns a new subclass of `impl_class` with ``__module__`` set to
    `mod_name`.  This insulates the underlying implementation from any patches
    we do in the shim module.

    Only directly-defined members of `impl_class` (not inherited ones) are
    inspected.  When a method's ``__doc__`` contains ``_skimage2`` /
    ``skimage2`` references that the doctest adapters would rewrite, a copy
    with the adapted docstring is set on the subclass.  The original class and
    its methods are never mutated.

    Parameters
    ----------
    impl_class : type
    shim_parents : sequence of type
        Shim classes of the bases of `impl_class`.
    class_doc : str or None
        Desired docstring for output class.
    mod_name : str
        Module name for the new class.

    Returns
    -------
    out : type
        Subclass of `impl_class` with adapted docstrings where needed, and
        ``out.__module__`` equal to `mod_name`.
    """
    out = _make_inheritor(impl_class, shim_parents, class_doc, mod_name)
    for name, member in vars(impl_class).items():
        if name.startswith('__') and name != '__init__':
            continue

        if isinstance(member, (staticmethod, classmethod)):
            unwrapped = member.__func__
        elif inspect.isroutine(member):
            unwrapped = member
        else:  # Not a (class-, static-, ordinary-) method.
            continue

        original_doc = getattr(unwrapped, '__doc__', None)
        if not original_doc:
            continue

        adapted_doc = _adapt_doctest_doc(original_doc)
        if adapted_doc == original_doc:
            continue

        new_func = _copy_callable(unwrapped, adapted_doc, mod_name)
        if isinstance(member, (staticmethod, classmethod)):
            new_func = type(member)(new_func)
        setattr(out, name, new_func)

    return out
