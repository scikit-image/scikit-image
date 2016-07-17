#!/usr/bin/env python

from __future__ import print_function
from skimage._build import cython
import os

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('detect', parent_package, top_path)
    config.add_data_dir('tests')

    # This function tries to create cpp files from the given .pyx files.  If
    # it fails, try to build with pre-generated .cpp files.

    cython(['cascade.pyx'], working_path=base_path)
    config.add_extension('cascade', sources=['cascade.cpp'],
                         include_dirs=[get_numpy_include_dirs()],
                         language="c++")
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup

    conf = configuration(top_path='').todict()

    setup(maintainer='scikit-image Developers',
          maintainer_email='scikit-image@googlegroups.com',
          description='Object detection framework',
          url='https://github.com/scikit-image/scikit-image',
          license='Modified BSD',
          **conf
          )
