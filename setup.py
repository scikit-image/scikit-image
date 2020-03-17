#! /usr/bin/env python

import os
import sys
import tempfile
import shutil
import builtins
import textwrap

import setuptools
from distutils.command.build_py import build_py
from distutils.command.sdist import sdist
from distutils.errors import CompileError, LinkError


DISTNAME = 'scikit-image'
DESCRIPTION = 'Image processing in Python'
MAINTAINER = 'Stefan van der Walt'
MAINTAINER_EMAIL = 'stefan@sun.ac.za'
URL = 'https://scikit-image.org'
LICENSE = 'Modified BSD'
DOWNLOAD_URL = 'https://scikit-image.org/docs/stable/install.html'
PROJECT_URLS = {
    "Bug Tracker": 'https://github.com/scikit-image/scikit-image/issues',
    "Documentation": 'https://scikit-image.org/docs/stable/',
    "Source Code": 'https://github.com/scikit-image/scikit-image'
}

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

if sys.version_info < (3, 6):

    error = """Python {py} detected.

scikit-image 0.16+ supports only Python 3.6 and above.

For Python 2.7, please install the 0.14.x Long Term Support release using:

 $ pip install 'scikit-image<0.15'
""".format(py='.'.join([str(v) for v in sys.version_info[:3]]))

    sys.stderr.write(error + "\n")
    sys.exit(1)

# This is a bit (!) hackish: we are setting a global variable so that the main
# skimage __init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet:
# the numpy distutils extensions that are used by scikit-image to recursively
# build the compiled extensions in sub-packages is based on the Python import
# machinery.
builtins.__SKIMAGE_SETUP__ = True


# Support for openmp

def openmp_build_ext():
    from numpy.distutils.command.build_ext import build_ext

    compile_flags = ['-fopenmp']
    link_flags = ['-fopenmp']

    code = """#include <omp.h>
    int main(int argc, char** argv) { return(0); }"""

    class ConditionalOpenMP(build_ext):

        def can_compile_link(self):

            cc = self.compiler
            fname = 'test.c'
            cwd = os.getcwd()
            tmpdir = tempfile.mkdtemp()

            try:
                os.chdir(tmpdir)
                with open(fname, 'wt') as fobj:
                    fobj.write(code)
                try:
                    objects = cc.compile([fname],
                                         extra_postargs=compile_flags)
                except CompileError:
                    return False
                try:
                    # Link shared lib rather then executable to avoid
                    # http://bugs.python.org/issue4431 with MSVC 10+
                    cc.link_shared_lib(objects, "testlib",
                                       extra_postargs=link_flags)
                except (LinkError, TypeError):
                    return False
            finally:
                os.chdir(cwd)
                shutil.rmtree(tmpdir)
            return True

        def build_extensions(self):
            """ Hook into extension building to check compiler flags """

            if self.can_compile_link():

                for ext in self.extensions:
                    ext.extra_compile_args += compile_flags
                    ext.extra_link_args += link_flags

            build_ext.build_extensions(self)

    return ConditionalOpenMP


with open('skimage/__init__.py') as fid:
    for line in fid:
        if line.startswith('__version__'):
            VERSION = line.strip().split()[-1][1:-1]
            break


def parse_requirements_file(filename):
    with open(filename) as fid:
        requires = [l.strip() for l in fid.readlines() if l]

    return requires


INSTALL_REQUIRES = parse_requirements_file('requirements/default.txt')
# The `requirements/extras.txt` file is explicitely omitted because
# it contains requirements that do not have wheels uploaded to pip
# for the platforms we wish to support.
extras_require = {
    dep: parse_requirements_file('requirements/' + dep + '.txt')
    for dep in ['docs', 'optional', 'test']
}

# requirements for those browsing PyPI
REQUIRES = [r.replace('>=', ' (>= ') + ')' for r in INSTALL_REQUIRES]
REQUIRES = [r.replace('==', ' (== ') for r in REQUIRES]
REQUIRES = [r.replace('[array]', '') for r in REQUIRES]


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True)

    config.add_subpackage('skimage')

    return config


if __name__ == "__main__":
    try:
        from numpy.distutils.core import setup
        extra = {'configuration': configuration}
    except ImportError:
        if len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
                                   sys.argv[1] in ('--help-commands',
                                                   '--version',
                                                   'clean',
                                                   'egg_info',
                                                   'install_egg_info',
                                                   'rotate')):
            # For these actions, NumPy is not required.
            #
            # They are required to succeed without Numpy for example when
            # pip is used to install scikit-image when Numpy is not yet
            # present in the system.
            from setuptools import setup
            extra = {}
        else:
            print(textwrap.dedent("""
                To install scikit-image from source, you will need NumPy.
                Install NumPy with pip using:

                  pip install numpy

                Or use your operating system package manager. For more
                details, see:

                   https://scikit-image.org/docs/stable/install.html
            """))
            sys.exit(1)

    setup(
        name=DISTNAME,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        url=URL,
        license=LICENSE,
        download_url=DOWNLOAD_URL,
        project_urls=PROJECT_URLS,
        version=VERSION,

        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: C',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3 :: Only',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
        ],
        install_requires=INSTALL_REQUIRES,
        requires=REQUIRES,
        extras_require=extras_require,
        python_requires='>=3.6',
        packages=setuptools.find_packages(exclude=['doc', 'benchmarks']),
        include_package_data=True,
        zip_safe=False,  # the package can run out of an .egg file

        entry_points={
            'console_scripts': ['skivi = skimage.scripts.skivi:main'],
        },

        cmdclass={'build_py': build_py,
                  'build_ext': openmp_build_ext(),
                  'sdist': sdist},
        **extra
    )
