#!/usr/bin/env python

import os
from skimage._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('filter', parent_package, top_path)
    config.add_data_dir('tests')

    cython(['_inpaint.pyx'], working_path=base_path)
    cython(['_ctmf.pyx'], working_path=base_path)
    cython(['_denoise_cy.pyx'], working_path=base_path)
    cython(['rank/core_cy.pyx'], working_path=base_path)
    cython(['rank/generic_cy.pyx'], working_path=base_path)
    cython(['rank/percentile_cy.pyx'], working_path=base_path)
    cython(['rank/bilateral_cy.pyx'], working_path=base_path)

    config.add_extension('_inpaint', sources=['_inpaint.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_ctmf', sources=['_ctmf.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_denoise_cy', sources=['_denoise_cy.c'],
                         include_dirs=[get_numpy_include_dirs(), '../_shared'])
    config.add_extension('rank._core8', sources=['rank/_core8.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('rank._core16', sources=['rank/_core16.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('rank._crank8', sources=['rank/_crank8.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension(
        'rank._crank8_percentiles', sources=['rank/_crank8_percentiles.c'],
        include_dirs=[get_numpy_include_dirs()])
    config.add_extension('rank._crank16', sources=['rank/_crank16.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension(
        'rank.bilateral_cy', sources=['rank/bilateral_cy.c'],
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
