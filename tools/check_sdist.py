#!/usr/bin/env python


import os
import subprocess
import sys

base_dir = os.path.join(os.path.dirname(__file__), '..')
os.chdir(base_dir)

p = subprocess.Popen("python setup.py sdist".split(),
                     stdout=subprocess.PIPE)
out, err = p.communicate()

data = out.decode('utf-8').split('\n')
data = [l for l in data if l.startswith('hard linking')]
data = [l.replace('hard linking ', '') for l in data]
data = ['./' + l.split(' ->')[0] for l in data]

ignore_exts = ['.pyc', '.so', '.o', '#', '~', '.gitignore', '.o.d']
ignore_dirs = ['./build', './dist', './tools', './doc', './viewer_examples',
               './downloads', './scikit_image.egg-info', './benchmarks']
ignore_files = ['./TODO.md', './README.md', './MANIFEST',
                './.gitignore', './.travis.yml', './.gitmodules',
                './.mailmap', './.coveragerc', './.appveyor.yml',
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

                if not (fn in data or fn in ignore_files):
                    missing.append(fn)

if missing:
    print('Missing from source distribution:\n')
    for m in missing:
        print('  ', m)

    print('\nPlease update MANIFEST.in')

    sys.exit(1)
else:
    print('All expected source files accounted for in sdist')
