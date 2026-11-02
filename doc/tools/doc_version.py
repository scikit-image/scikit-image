"""Get __version__ string from __init__.py"""

from pathlib import Path
import re


VERSION_RE = re.compile(
    r'''
    ^  # No indent
    __version__
    \s*=\s*
    (["\'])
    (?P<version>.*?)
    \1  # Match initial single / double quote.
    \s*$''',
    flags=re.MULTILINE | re.VERBOSE,
)


HERE = Path(__file__).resolve().parent


def get_version():
    init_pth = HERE.parent.parent / 'src' / 'skimage' / '__init__.py'
    if v_match := VERSION_RE.search(init_pth.read_text()):
        return v_match.group('version')
    return 'vUndefined'
