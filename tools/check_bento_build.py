#!/usr/bin/env python
"""
Check that Cython extensions in setup.py files match those in bento.info.
"""
import os
import re
import sys


RE_CYTHON = re.compile("config.add_extension\(\s*['\"]([\S]+)['\"]")

BENTO_TEMPLATE = """
    Extension: {module_path}
        Sources:
            {dir_path}.pyx"""


def each_setup_in_pkg(top_dir):
    """Yield path and file object for each setup.py file"""
    for dir_path, dir_names, filenames in os.walk(top_dir):
        for fname in filenames:
            if fname == 'setup.py':
                with open(os.path.join(dir_path, 'setup.py')) as f:
                    yield dir_path, f


def each_cy_in_setup(top_dir):
    """Yield path for each cython extension package's setup file."""
    for dir_path, f in each_setup_in_pkg(top_dir):
        text = f.read()
        match = RE_CYTHON.findall(text)
        if match:
            for cy_file in match:
                # if cython files in different directory than setup.py
                if '.' in cy_file:
                    parts = cy_file.split('.')
                    cy_file = parts[-1]
                    # Don't overwrite dir_path for subsequent iterations.
                    path = os.path.join(dir_path, *parts[:-1])
                else:
                    path = dir_path
                full_path = os.path.join(path, cy_file)
                yield full_path


def each_cy_in_bento(bento_file='bento.info'):
    """Yield path for each cython extension in bento info file."""
    with open(bento_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('Extension:'):
                path = line.lstrip('Extension:').strip()
                yield path


def remove_common_extensions(cy_bento, cy_setup):
    # normalize so that cy_setup and cy_bento have the same separator
    cy_setup = set(ext.replace('/', '.') for ext in cy_setup)
    cy_setup_diff = cy_setup.difference(cy_bento)
    cy_setup_diff = set(ext.replace('.', '/') for ext in cy_setup_diff)
    cy_bento_diff = cy_bento.difference(cy_setup)
    return cy_bento_diff, cy_setup_diff


def print_results(cy_bento, cy_setup):
    def info(text):
        print('')
        print(text)
        print('-' * len(text))

    if not (cy_bento or cy_setup):
        print("bento.info and setup.py files match.")

    if cy_bento:
        info("Extensions found in 'bento.info' but not in any 'setup.py:")
        print('\n'.join(cy_bento))


    if cy_setup:
        info("Extensions found in a 'setup.py' but not in any 'bento.info:")
        print('\n'.join(cy_setup))
        info("Consider adding the following to the 'bento.info' Library:")
        for dir_path in cy_setup:
            module_path = dir_path.replace('/', '.')
            print(BENTO_TEMPLATE.format(module_path=module_path,
                                        dir_path=dir_path))


if __name__ == '__main__':
    # All cython extensions defined in 'setup.py' files.
    cy_setup = set(each_cy_in_setup('skimage'))

    # All cython extensions defined 'bento.info' file.
    cy_bento = set(each_cy_in_bento())

    cy_bento, cy_setup = remove_common_extensions(cy_bento, cy_setup)
    print_results(cy_bento, cy_setup)

    if cy_setup or cy_bento:
        sys.exit(1)
