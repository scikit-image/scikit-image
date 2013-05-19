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
VERSION             = '0.9dev'
PYTHON_VERSION      = (2, 5)
DEPENDENCIES        = {
                        'numpy': (1, 6),
                        'Cython': (0, 15),
                      }


import os
import sys
import setuptools
import re
from numpy.distutils.core import setup
from numpy.distutils.exec_command import exec_command, find_executable
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


def get_package_version(package):
    version = []
    for version_attr in ('version', 'VERSION', '__version__'):
        if hasattr(package, version_attr) \
                and isinstance(getattr(package, version_attr), str):
            version_info = getattr(package, version_attr, '')
            for part in re.split('\D+', version_info):
                try:
                    version.append(int(part))
                except ValueError:
                    pass
    return tuple(version)


def check_requirements():
    if sys.version_info < PYTHON_VERSION:
        raise SystemExit('You need Python version %d.%d or later.' \
                         % PYTHON_VERSION)

    for package_name, min_version in DEPENDENCIES.items():
        dep_error = False
        try:
            package = __import__(package_name)
        except ImportError:
            dep_error = True
        else:
            package_version = get_package_version(package)
            if min_version > package_version:
                dep_error = True

        if dep_error:
            raise ImportError('You need `%s` version %d.%d or later.' \
                              % ((package_name, ) + min_version))

# uses searching from waf docs: 
# http://docs.waf.googlecode.com/git/book_16/single.html#_download_and_installation

def waflib_exists():
    if os.environ.get('WAFDIR'):
        return True
    if 'waf' in os.listdir(os.getcwd()) and 'waflib' in os.listdir(os.path.join(os.getcwd(),'waflib')):
        return True
    if len(re.findall(re.findall('[.]waf-[0-9][.][0-9]-version',os.listdir(os.getcwd()))))>0:
        return True
    return False

if __name__ == "__main__":

    check_requirements()

    write_version_py()

    # check for bento installation
    bento_path = find_executable('bentomaker')
    if bento_path and waflib_exists():
        exec_command(bento_path+' install')
    else:
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

            packages=setuptools.find_packages(exclude=['doc']),
            include_package_data=True,
            zip_safe=False, # the package can run out of an .egg file

            entry_points={
                'console_scripts': ['skivi = skimage.scripts.skivi:main'],
            },

            cmdclass={'build_py': build_py},
        )
