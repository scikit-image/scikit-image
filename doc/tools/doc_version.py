"""Read the package version from the source tree."""

from pathlib import Path

VERSION_FILE = Path(__file__).resolve().parents[2] / 'src' / '_skimage2' / '_version.py'


def get_version():
    """Return the ``__version__`` string defined in ``_skimage2/_version.py``."""
    for line in VERSION_FILE.read_text().splitlines():
        if line.startswith('__version__ ='):
            return line.split('=', 1)[1].strip().strip('\'"')
    raise RuntimeError(f'No `__version__ = ` line in {VERSION_FILE}')
