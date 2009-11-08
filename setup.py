#! /usr/bin/env python

descr   = """Image Processing SciKit

Image processing algorithms for SciPy, including IO, morphology, filtering,
warping, color manipulation, object detection, etc.

Please refer to the online documentation at
http://stefanv.github.com/scikits.image
"""

DISTNAME            = 'scikits.image'
DESCRIPTION         = 'Image processing routines for SciPy'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Stefan van der Walt',
MAINTAINER_EMAIL    = 'stefan@sun.ac.za',
URL                 = 'http://stefanv.github.com/scikits.image'
LICENSE             = 'Modified BSD'
DOWNLOAD_URL        = 'http://github.com/stefanv/scikits.image'
VERSION             = '0.2dev'

import os
import setuptools
from numpy.distutils.core import setup

def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path,
                           namespace_packages=['scikits'])

    config.add_subpackage('scikits')
    config.add_subpackage(DISTNAME)
    config.add_data_files('scikits/__init__.py')
    config.add_data_dir('scikits/image/data')

    return config

def write_version_py(filename='scikits/image/version.py'):
    template = """# THIS FILE IS GENERATED FROM THE SCIKITS.IMAGE SETUP.PY
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
            [ 'Development Status :: 1 - Planning',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: BSD License',
              'Topic :: Scientific/Engineering'],

        configuration=configuration,
        install_requires=[],
        namespace_packages=['scikits'],
        packages=setuptools.find_packages(),
        include_package_data=True,
        zip_safe=False, # the package can run out of an .egg file

        entry_points={
            'console_scripts': [
                'scivi = scikits.image.scripts.scivi:main']
            },
        )
