#! /usr/bin/env python

descr = """Image Processing SciKit

Image processing algorithms for SciPy, including IO, morphology, filtering,
warping, color manipulation, object detection, etc.

Please refer to the online documentation at
http://scikit-image.org/
"""

DISTNAME = 'scikit-image'
DESCRIPTION = 'Image processing routines for SciPy'
LONG_DESCRIPTION = descr
MAINTAINER = 'Stefan van der Walt'
MAINTAINER_EMAIL = 'stefan@sun.ac.za'
URL = 'http://scikit-image.org'
LICENSE = 'Modified BSD'
DOWNLOAD_URL = 'http://github.com/scikit-image/scikit-image'

import os
import sys

import setuptools
from distutils.command.build_py import build_py
from distutils.command.sdist import sdist

if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins

# This is a bit (!) hackish: we are setting a global variable so that the main
# skimage __init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet:
# the numpy distutils extensions that are used by scikit-image to recursively
# build the compiled extensions in sub-packages is based on the Python import
# machinery.
builtins.__SKIMAGE_SETUP__ = True


with open('skimage/__init__.py') as fid:
    for line in fid:
        if line.startswith('__version__'):
            VERSION = line.strip().split()[-1][1:-1]
            break

with open('requirements.txt') as fid:
    INSTALL_REQUIRES = [l.strip() for l in fid.readlines() if l]

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
    config.add_data_dir('skimage/data')

    return config


if __name__ == "__main__":
    try:
        from numpy.distutils.core import setup
        extra = {'configuration': configuration}
        # Do not try and upgrade larger dependencies
        for lib in ['scipy', 'numpy', 'matplotlib', 'pillow']:
            try:
                __import__(lib)
                INSTALL_REQUIRES = [i for i in INSTALL_REQUIRES
                                    if lib not in i]
            except ImportError:
                pass
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
            print('To install scikit-image from source, you will need numpy.\n' +
                  'Install numpy with pip:\n' +
                  'pip install numpy\n'
                  'Or use your operating system package manager. For more\n' +
                  'details, see http://scikit-image.org/docs/stable/install.html')
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
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
        ],
        install_requires=INSTALL_REQUIRES,
        requires=REQUIRES,
        packages=setuptools.find_packages(exclude=['doc']),
        include_package_data=True,
        zip_safe=False,  # the package can run out of an .egg file

        entry_points={
            'console_scripts': ['skivi = skimage.scripts.skivi:main'],
        },

        cmdclass={'build_py': build_py,
                  'sdist': sdist},
        **extra
    )
