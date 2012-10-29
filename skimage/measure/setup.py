#!/usr/bin/env python

from skimage._build import cython

import os
base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('measure', parent_package, top_path)
    config.add_data_dir('tests')

    cython(['_find_contours.pyx'], working_path=base_path)
    cython(['_moments.pyx'], working_path=base_path)

    config.add_extension('_find_contours', sources=['_find_contours.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_moments', sources=['_moments.c'],
                         include_dirs=[get_numpy_include_dirs()])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='scikit-image Developers',
          maintainer_email='scikit-image@googlegroups.com',
          description='Graph-based Image-processing Algorithms',
          url='https://github.com/scikit-image/scikit-image',
          license='Modified BSD',
          **(configuration(top_path='').todict())
          )
