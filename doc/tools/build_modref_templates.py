#!/usr/bin/env python
"""Script to auto-generate our API docs."""

import sys

from packaging import version as _version

# local imports
from apigen import ApiDocWriter


# *****************************************************************************


def abort(error):
    print(f'*WARNING* API documentation not generated: {error}')
    exit()


if __name__ == '__main__':
    package = 'skimage'

    # Check that the 'image' package is available. If not, the API
    # documentation is not (re)generated and existing API documentation
    # sources will be used.

    try:
        __import__(package)
    except ImportError:
        abort("Can not import skimage")

    module = sys.modules[package]

    # Check that the source version is equal to the installed
    # version. If the versions mismatch the API documentation sources
    # are not (re)generated. This avoids automatic generation of documentation
    # for older or newer versions if such versions are installed on the system.

    # exclude any appended git hash and date
    installed_version = _version.parse(module.__version__.split('+git')[0])

    source_lines = open('../src/skimage/__init__.py').readlines()
    version = 'vUndefined'
    for l in source_lines:
        if l.startswith('__version__ = '):
            source_version = _version.parse(l.split("'")[1])
            break

    if source_version != installed_version:
        abort("Installed version does not match source version")

    outdir = 'source/api'
    docwriter = ApiDocWriter(package)
    docwriter.package_skip_patterns += [
        r'\.fixes$',
        r'\.externals$',
        r'filter$',
    ]
    docwriter.write_api_docs(outdir)
    docwriter.write_index(outdir, 'api', relative_to='source/api')

    if len(docwriter.written_modules) <= 1:
        msg = (
            f"expected more modules, only wrote files for: "
            f"{docwriter.written_modules!r}"
        )
        raise RuntimeWarning(msg)
    else:
        print(f'{len(docwriter.written_modules)} files written')

    package2 = '_skimage2'
    try:
        __import__(package2)
    except ImportError:
        abort("Can not import skimage2")

    outdir2 = 'source/api2'
    docwriter2 = ApiDocWriter(package2, display_package_name='skimage2')
    docwriter2.package_skip_patterns += [r'\.tests$']
    docwriter2.module_preamble = (
        ".. warning::\n\n"
        "   This module is part of the **experimental** ``skimage2`` namespace"
        " and is subject to change without notice.\n"
        "   Do not use it in production code.\n"
        "   See the :ref:`migration guide <skimage2-migration>` for more details."
    )
    docwriter2.write_api_docs(outdir2)
    docwriter2.write_index(
        outdir2,
        'api2',
        relative_to='source/api2',
        title="Experimental: skimage2 API reference",
        include_license=False,
        preamble=(
            ".. warning::\n\n"
            "   ``skimage2`` is **experimental** and subject to change without"
            " notice.\n"
            "   The API may be altered or removed in any future release.\n"
            "   Do not use it in production code.\n"
            "   See the :ref:`migration guide <skimage2-migration>` for more details."
        ),
    )

    if len(docwriter2.written_modules) <= 1:
        msg = (
            f"expected more _skimage2 modules, only wrote files for: "
            f"{docwriter2.written_modules!r}"
        )
        raise RuntimeWarning(msg)
    else:
        print(f'{len(docwriter2.written_modules)} _skimage2 files written')
