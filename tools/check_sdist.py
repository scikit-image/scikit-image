#!/usr/bin/env python3
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
import tarfile

parser = ArgumentParser(description='Check a created sdist')
parser.add_argument('sdist_name', type=str, nargs=1,
                    help='The name of the sdist file to check')
args = parser.parse_args()
sdist_name = args.sdist_name[0]

with tarfile.open(sdist_name) as tar:
    members = tar.getmembers()

# The very first item contains the name of the archive
top_parent = Path(members[0].name)

filenames = ['./' + str(Path(m.name).relative_to(top_parent))
             for m in members[1:]]

ignore_exts = ['.pyc', '.so', '.o', '#', '~', '.gitignore', '.o.d']
ignore_dirs = ['./build', './dist', './tools', './doc', './viewer_examples',
               './downloads', './scikit_image.egg-info', './benchmarks']
ignore_files = ['./TODO.md', './README.md', './MANIFEST',
                './.gitignore', './.travis.yml', './.gitmodules',
                './.mailmap', './.coveragerc', './azure-pipelines.yml',
                './.pep8speaks.yml', './asv.conf.json',
                './.codecov.yml',
                './skimage/filters/rank/README.rst', './.meeseeksdev.yml']

# These docstring artifacts are hard to avoid without adding noise to the
# docstrings. They typically show up if you run the whole test suite in the
# build directory.
docstring_artifacts = ['./temp.tif', './save-demo.jpg']
ignore_files = ignore_files + docstring_artifacts


missing = []
for root, dirs, files in os.walk('./'):
    for d in ignore_dirs:
        if root.startswith(d):
            break
    else:

        if root.startswith('./.'):
            continue

        for fn in files:
            for ext in ignore_exts:
                if fn.endswith(ext):
                    break
            else:
                fn = os.path.join(root, fn)

                if not (fn in filenames or fn in ignore_files):
                    missing.append(fn)

if missing:
    print('Missing from source distribution:\n')
    for m in missing:
        print('  ', m)

    print('\nPlease update MANIFEST.in')

    sys.exit(1)
else:
    print('All expected source files accounted for in sdist')
