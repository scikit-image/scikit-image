"""
Check that Cython extensions in setup.py files match those in bento.info.
"""
import os
import re


RE_CYTHON = re.compile("config.add_extension\(['\"]([\S]+)['\"]")

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
    """Yield path and name for each cython extension package's setup file."""
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
                yield full_path, cy_file


def each_cy_in_bento(bento_file='bento.info'):
    """Yield path and name for each cython extension in bento info file."""
    with open(bento_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('Extension:'):
                parts = line.split('.')
                ext_name = parts[-1]
                path = line.lstrip('Extension:').strip()
                yield path, ext_name


def remove_common_extensions(cy_bento, cy_setup):
    for ext_name in cy_bento.keys():
        if ext_name in cy_setup:
            spath = cy_setup.pop(ext_name)
            bpath = cy_bento.pop(ext_name)
            if not spath.replace(os.path.sep, '.') == bpath:
                print "Mismatched paths:"
                print "    setup.py:  ", spath
                print "    bento.info:", bpath

def print_results(cy_bento, cy_setup):
    def info(text):
        print
        print(text)
        print('-' * len(text))

    print # blank line; just for aesthetics

    if cy_bento:
        info("The following extensions in 'bento.info' were not found:")
        print('\n'.join(cy_bento.keys()))


    if cy_setup:
        info("The following cython files exist but were not in 'bento.info':")
        print('\n'.join(cy_setup))
        info("Consider adding the following to the 'bento.info' Library:")
        for ext_name, dir_path in cy_setup.iteritems():
            print BENTO_TEMPLATE.format(module_path=dir_path.replace('/', '.'),
                                        dir_path=dir_path)

if __name__ == '__main__':
    # All cython extensions defined in 'setup.py' files.
    cy_setup = dict((ext, path) for path, ext in each_cy_in_setup('skimage'))

    # All cython extensions defined 'bento.info' file.
    cy_bento = dict((ext, path) for path, ext in each_cy_in_bento())

    remove_common_extensions(cy_bento, cy_setup)
    print_results(cy_bento, cy_setup)
