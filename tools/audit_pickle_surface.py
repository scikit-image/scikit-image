#!/usr/bin/env python3
"""Audit pickle compatibility coverage for the skimage megamove.

Lists registered public paths, symbols passed to pickle helpers in tests, and
stateful ``_skimage2`` classes that are not registered yet. With ``--strict``,
exits with status 1 when unregistered stateful classes are found.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
SKIMAGE2_ROOT = SRC_ROOT / '_skimage2'
TESTS_ROOT = REPO_ROOT / 'tests'

_STATEFUL_MARKERS = ('__getstate__', '__setstate__', '__reduce__', '__reduce_ex__')


def _iter_python_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob('*.py'))


def _impl_name(ref) -> str:
    return ref.module.replace('skimage.', '_skimage2.', 1) + f'.{ref.qualname}'


def _find_stateful_classes() -> set[str]:
    candidates: set[str] = set()
    for path in _iter_python_files(SKIMAGE2_ROOT):
        source = path.read_text(encoding='utf-8')
        if not any(marker in source for marker in _STATEFUL_MARKERS):
            continue
        rel = path.relative_to(SRC_ROOT).with_suffix('')
        module = '.'.join(rel.parts)
        for node in ast.parse(source, filename=str(path)).body:
            if isinstance(node, ast.ClassDef):
                candidates.add(f'{module}.{node.name}')
    return candidates


def _pickle_dump_args(source: str) -> set[str]:
    if 'pickle.dumps' not in source and 'pickle_dumps' not in source:
        return set()
    args: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == 'pickle_dumps':
            args.add(ast.unparse(node.args[0]))
        elif (
            isinstance(func, ast.Attribute)
            and func.attr == 'dumps'
            and isinstance(func.value, ast.Name)
            and func.value.id == 'pickle'
        ):
            args.add(ast.unparse(node.args[0]))
    return args


def _load_registered_refs():
    sys.path.insert(0, str(SRC_ROOT))
    from skimage._pickle_compat import registered_pickle_refs

    return registered_pickle_refs()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Fail when stateful _skimage2 classes are not registered.',
    )
    args = parser.parse_args(argv)

    refs = _load_registered_refs()
    registered_impl = {_impl_name(ref) for ref in refs}
    unregistered = sorted(_find_stateful_classes() - registered_impl)

    print('Registered pickle paths:')
    for ref in refs:
        print(f'  - {ref}')

    print('\nPickle test expressions:')
    expressions: set[str] = set()
    for path in _iter_python_files(TESTS_ROOT):
        expressions.update(_pickle_dump_args(path.read_text(encoding='utf-8')))
    for expr in sorted(expressions):
        print(f'  - {expr}')

    if unregistered:
        print('\nStateful _skimage2 classes without registration:')
        for name in unregistered:
            print(f'  - {name}')
    else:
        print('\nAll stateful _skimage2 classes are registered.')

    return 1 if args.strict and unregistered else 0


if __name__ == '__main__':
    raise SystemExit(main())
