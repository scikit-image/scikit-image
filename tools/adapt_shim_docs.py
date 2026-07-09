#!/usr/bin/env python3
"""Add bind_namespace() calls to skimage shim modules."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIM_ROOT = REPO_ROOT / 'src' / 'skimage'

SKIP_FILES = {
    SHIM_ROOT / 'conftest.py',
    SHIM_ROOT / '_migration.py',
    SHIM_ROOT / 'util' / 'lookfor.py',
}

SKIP_NAMES = {'__init__.py', '__init__.pyi'}

BIND_BLOCK = '''\
from skimage._docutils import bind_namespace

bind_namespace(globals())
'''

IMPORT_DOC_RE = re.compile(
    r'^from _skimage2[^\n]* import __doc__[^\n]*\n', re.MULTILINE
)
HAS_BIND_RE = re.compile(r'bind_namespace\s*\(')


def adapt_shim(path: Path) -> bool:
    if path in SKIP_FILES or path.name in SKIP_NAMES:
        return False
    text = path.read_text()
    if 'from _skimage2' not in text and 'import _skimage2' not in text:
        return False
    if HAS_BIND_RE.search(text):
        return False

    text = IMPORT_DOC_RE.sub('', text)
    if not text.endswith('\n'):
        text += '\n'
    text += '\n' + BIND_BLOCK
    path.write_text(text)
    return True


def main():
    changed = []
    for path in sorted(SHIM_ROOT.rglob('*.py')):
        if adapt_shim(path):
            changed.append(path.relative_to(REPO_ROOT))
    print(f'Updated {len(changed)} shim files')
    for rel in changed:
        print(f'  {rel}')


if __name__ == '__main__':
    main()
