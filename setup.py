#! /usr/bin/env python

descr   = """Image Processing SciKit

Image processing algorithms for SciPy, including IO, morphology, filtering,
warping, color manipulation, object detection, etc.

Please refer to the online documentation at
http://scikits-image.org/
"""

DISTNAME            = 'scikits-image'
DESCRIPTION         = 'Image processing routines for SciPy'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Stefan van der Walt'
MAINTAINER_EMAIL    = 'stefan@sun.ac.za'
URL                 = 'http://scikits-image.org'
LICENSE             = 'Modified BSD'
DOWNLOAD_URL        = 'http://github.com/scikits-image/scikits-image'
VERSION             = '0.5'

import os
import setuptools
from numpy.distutils.core import setup
try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    from distutils.command.build_py import build_py

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

def write_version_py(filename='skimage/version.py'):
    template = """# THIS FILE IS GENERATED FROM THE SKIMAGE SETUP.PY
version='%s'
"""

    vfile = open(os.path.join(os.path.dirname(__file__),
                              filename), 'w')

    try:
        vfile.write(template % VERSION)
    finally:
        vfile.close()

if __name__ == "__main__":
    write_version_py()

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

        classifiers =
            [ 'Development Status :: 4 - Beta',
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
        install_requires=[],
        packages=setuptools.find_packages(),
        include_package_data=True,
        zip_safe=False, # the package can run out of an .egg file

        entry_points={
            'console_scripts': [
                'skivi = skimage.scripts.skivi:main']
            },

        cmdclass={'build_py': build_py},
        )
