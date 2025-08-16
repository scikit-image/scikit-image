#!/usr/bin/env python3
"""Extract version number from __init__.py"""

import os


def git_version(version):
    """Append last commit date and hash to version, if available"""
    import subprocess
    import os.path

    git_hash = ''
    try:
        p = subprocess.Popen(
            ['git', 'log', '-1', '--format="%H %aI"'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(__file__),
        )
    except FileNotFoundError:
        pass
    else:
        out, err = p.communicate()
        if p.returncode == 0:
            git_hash, git_date = (
                out.decode('utf-8')
                .strip()
                .replace('"', '')
                .split('T')[0]
                .replace('-', '')
                .split()
            )

            version += f'+git{git_date}.{git_hash[:7]}'

    return version


ski_init = os.path.join(os.path.dirname(__file__), '../__init__.py')

data = open(ski_init).readlines()
version_line = next(line for line in data if line.startswith('__version__ ='))

version = version_line.strip().split(' = ')[1].replace('"', '').replace("'", '')

if 'dev' in version:
    version = git_version(version)

print(version)
