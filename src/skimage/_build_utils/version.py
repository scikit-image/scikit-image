#!/usr/bin/env python3

"""Determine and print version number.

Used in top level ``meson.build``.
"""

import subprocess
from pathlib import Path


def version_from_init():
    """Extract version string from ``skimage/__init__.py``."""
    skimage_init = Path(__file__).parent / '../__init__.py'
    assert skimage_init.is_file()

    with skimage_init.open("r") as file:
        data = file.readlines()

    version_line = next(line for line in data if line.startswith('__version__ ='))
    version = version_line.strip().split(' = ')[1].replace('"', '').replace("'", '')
    return version


def append_git_revision_and_date(version):
    """Try to append last commit date and hash to version.

    Appends nothing if the current working directory is outside a git
    repository.
    """
    try:
        result = subprocess.run(
            ['git', 'log', '-1', '--format="%H %aI"'],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        pass
    else:
        git_hash, git_date = (
            result.stdout.strip()
            .replace('"', '')
            .split('T')[0]
            .replace('-', '')
            .split()
        )
        version += f'+git{git_date}.{git_hash[:7]}'
    return version


if __name__ == "__main__":
    version = version_from_init()
    if 'dev' in version:
        version = append_git_revision_and_date(version)
    print(version)
