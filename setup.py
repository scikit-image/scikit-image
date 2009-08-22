#! /usr/bin/env python

descr   = """Image Processing SciKit

Provide image processing capabilities to SciPy, including:

- Image IO without PIL dependencies
- Image warping (wrappers based on ndimage)
- Connected components
- Color-space manipulations
- Linear space-invariant filters
- Hough transform
- Shortest paths
- Grey-level co-occurrence matrices
- Edge detection
- Image collections

"""

import os
import sys

DISTNAME            = 'scikits.image'
DESCRIPTION         = 'Image processing routines for SciPy'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Stefan van der Walt',
MAINTAINER_EMAIL    = 'stefan@sun.ac.za',
URL                 = 'http://github.com/stefanv/scikits.image'
LICENSE             = 'Modified BSD'
DOWNLOAD_URL        = URL
VERSION             = '0.1'

import setuptools
from numpy.distutils.core import setup

def configuration(parent_package='', top_path=None, package_name=DISTNAME):
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(package_name, parent_package, top_path,
                           version = VERSION,
                           maintainer  = MAINTAINER,
                           maintainer_email = MAINTAINER_EMAIL,
                           description = DESCRIPTION,
                           license = LICENSE,
                           url = URL,
                           download_url = DOWNLOAD_URL,
                           long_description = LONG_DESCRIPTION)

    return config

if __name__ == "__main__":
    setup(configuration = configuration,
        install_requires = 'numpy',
        namespace_packages = ['scikits'],
        packages = setuptools.find_packages(),
        include_package_data = True,
        #test_suite="tester", # for python setup.py test
        zip_safe = True, # the package can run out of an .egg file
        classifiers =
            [ 'Development Status :: 1 - Planning',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: BSD License',
              'Topic :: Scientific/Engineering'])

