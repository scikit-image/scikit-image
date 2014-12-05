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
VERSION             = '0.11dev'
PYTHON_VERSION      = (2, 6)

import re
import os
import sys

import setuptools
from distutils.command.build_py import build_py
from distutils.version import LooseVersion


# These are manually checked.
# These packages are sometimes installed outside of the setuptools scope
DEPENDENCIES = {}
with open('requirements.txt', 'rb') as fid:
    data = fid.read().decode('utf-8', 'replace')
for line in data.splitlines():
    pkg, _, version_info = line.replace('==', '>=').partition('>=')
    # Only require Cython if we have a developer checkout
    if pkg.lower() == 'cython' and not VERSION.endswith('dev'):
        continue
    DEPENDENCIES[str(pkg).lower()] = str(version_info)


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


def get_package_version(package):
    for version_attr in ('__version__', 'VERSION', 'version'):
        version_info = getattr(package, version_attr, None)
        if version_info and str(version_attr) == version_attr:
            return str(version_info)


def check_requirements():
    if sys.version_info < PYTHON_VERSION:
        raise SystemExit('You need Python version %d.%d or later.' \
                         % PYTHON_VERSION)
    for (package_name, min_version) in DEPENDENCIES.items():
        if package_name == 'cython':
            package_name = 'Cython'
        dep_error = ''
        if package_name.lower() == 'pillow':
            package_name = 'PIL.Image'
            min_version = '1.1.7'
        try:
            package = __import__(package_name,
                fromlist=[package_name.rpartition('.')[0]])
        except ImportError:
            dep_error = ('You need `%s` version %s or later.'
                         % (package_name, min_version))
        else:
            if package_name == 'PIL':
                package_version = package.PILLOW_VERSION
            else:
                package_version = get_package_version(package)

            if LooseVersion(min_version) > LooseVersion(package_version):
                dep_error = ('You need `%s` version %s or later,'
                             'found version %s.'
                             % (package_name, min_version,
                                package_version))
        if dep_error:
            raise ImportError(dep_error)


if __name__ == "__main__":

    check_requirements()

    write_version_py()

    from numpy.distutils.core import setup
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
        install_requires=[
            "six>=%s" % DEPENDENCIES['six']
        ],
        packages=setuptools.find_packages(exclude=['doc']),
        include_package_data=True,
        zip_safe=False,  # the package can run out of an .egg file

        entry_points={
            'console_scripts': ['skivi = skimage.scripts.skivi:main'],
        },

        cmdclass={'build_py': build_py},
    )
