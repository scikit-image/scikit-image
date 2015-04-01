#! /usr/bin/env python

descr = """Image Processing SciKit

Image processing algorithms for SciPy, including IO, morphology, filtering,
warping, color manipulation, object detection, etc.

Please refer to the online documentation at
http://scikit-image.org/
"""

DISTNAME            = 'scikit-image'
DESCRIPTION         = 'Image processing routines for SciPy'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Stefan van der Walt'
MAINTAINER_EMAIL    = 'stefan@sun.ac.za'
URL                 = 'http://scikit-image.org'
LICENSE             = 'Modified BSD'
DOWNLOAD_URL        = 'http://github.com/scikit-image/scikit-image'

import os

import setuptools
from distutils.command.build_py import build_py


with open('skimage/__init__.py') as fid:
    for line in fid:
        if line.startswith('__version__'):
            VERSION = line.strip().split()[-1][1:-1]
            break

with open('requirements.txt') as fid:
    INSTALL_REQUIRES = [l.strip() for l in fid.readlines() if l]

# development versions do not have the cythonized files
if VERSION.endswith('dev'):
    SETUP_REQUIRES = [r for r in INSTALL_REQUIRES if r.startswith('cython')]
else:
    INSTALL_REQUIRES = [r for r in INSTALL_REQUIRES
                                         if not r.startswith('cython')]
    SETUP_REQUIRES = []


# list requirements for PyPI
REQUIRES = [r.replace('>=', ' (>= ') + ')'
                      for r in INSTALL_REQUIRES + SETUP_REQUIRES]
REQUIRES = [r.replace('==', ' (== ') for r in REQUIRES]


# do not attempt to install numpy and scipy until they have eggs available
INSTALL_REQUIRES = [r for r in INSTALL_REQUIRES
                                     if not r.startswith(('scipy', 'numpy'))]


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')

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
    # purposely fail loudly if numpy or scipy are not available
    from numpy.distutils.core import setup
    import scipy

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

        configuration=configuration,
        setup_requires=SETUP_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        requires=REQUIRES,
        packages=setuptools.find_packages(exclude=['doc']),
        include_package_data=True,
        zip_safe=False,  # the package can run out of an .egg file

        entry_points={
            'console_scripts': ['skivi = skimage.scripts.skivi:main'],
        },

        cmdclass={'build_py': build_py},
    )
