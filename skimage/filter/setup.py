#!/usr/bin/env python

import os
from skimage._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('filter', parent_package, top_path)
    config.add_data_dir('tests')

    cython(['_ctmf.pyx'], working_path=base_path)
    cython(['_denoise.pyx'], working_path=base_path)
    cython(['rank/_core8.pyx'], working_path=base_path)
    cython(['rank/_core16.pyx'], working_path=base_path)
    cython(['rank/_crank8.pyx'], working_path=base_path)
    cython(['rank/_crank8_percentiles.pyx'], working_path=base_path)
    cython(['rank/_crank16.pyx'], working_path=base_path)
    cython(['rank/_crank16_percentiles.pyx'], working_path=base_path)
    cython(['rank/_crank16_bilateral.pyx'], working_path=base_path)
    cython(['rank/rank.pyx'], working_path=base_path)
    cython(['rank/percentile_rank.pyx'], working_path=base_path)
    cython(['rank/bilateral_rank.pyx'], working_path=base_path)

    config.add_extension('_ctmf', sources=['_ctmf.c'],
        include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_denoise', sources=['_denoise.c'],
        include_dirs=[get_numpy_include_dirs(), '../_shared'])
    config.add_extension('rank/_core8', sources=['rank/_core8.c'],
        include_dirs=[get_numpy_include_dirs()])
    config.add_extension('rank/_core16', sources=['rank/_core16.c'],
        include_dirs=[get_numpy_include_dirs()])
    config.add_extension('rank/_crank8', sources=['rank/_crank8.c'],
        include_dirs=[get_numpy_include_dirs()])
    config.add_extension(
        'rank/_crank8_percentiles', sources=['rank/_crank8_percentiles.c'],
        include_dirs=[get_numpy_include_dirs()])
    config.add_extension('rank/_crank16', sources=['rank/_crank16.c'],
        include_dirs=[get_numpy_include_dirs()])
    config.add_extension(
        'rank/_crank16_percentiles', sources=['rank/_crank16_percentiles.c'],
        include_dirs=[get_numpy_include_dirs()])
    config.add_extension(
        'rank/_crank16_bilateral', sources=['rank/_crank16_bilateral.c'],
        include_dirs=[get_numpy_include_dirs()])
    config.add_extension(
        'rank/rank', sources=['rank/rank.c'],
        include_dirs=[get_numpy_include_dirs()])
    config.add_extension(
        'rank/percentile_rank', sources=['rank/percentile_rank.c'],
        include_dirs=[get_numpy_include_dirs()])
    config.add_extension(
        'rank/bilateral_rank', sources=['rank/bilateral_rank.c'],
        include_dirs=[get_numpy_include_dirs()])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='scikit-image Developers',
          author='scikit-image Developers',
          maintainer_email='scikit-image@googlegroups.com',
          description='Filters',
          url='https://github.com/scikit-image/scikit-image',
          license='SciPy License (BSD Style)',
          **(configuration(top_path='').todict())
          )
