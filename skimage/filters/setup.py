#!/usr/bin/env python

import os
from skimage._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('filters', parent_package, top_path)

    cython(['rank/_core_cy.pyx',
            'rank/_core_cy_3d.pyx',
            'rank/_generic_cy.pyx',
            'rank/_percentile_cy.pyx',
            'rank/_bilateral_cy.pyx',
            '_multiotsu.pyx'], working_path=base_path)

    config.add_extension('rank._core_cy', sources=['rank/_core_cy.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('rank._core_cy_3d', sources=['rank/_core_cy_3d.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_multiotsu', sources=['_multiotsu.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('rank._generic_cy', sources=['rank/_generic_cy.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension(
        'rank._percentile_cy', sources=['rank/_percentile_cy.c'],
        include_dirs=[get_numpy_include_dirs()])
    config.add_extension(
        'rank._bilateral_cy', sources=['rank/_bilateral_cy.c'],
        include_dirs=[get_numpy_include_dirs()])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='scikit-image Developers',
          author='scikit-image Developers',
          maintainer_email='skimage@discuss.scientific-python.org',
          description='Filters',
          url='https://github.com/scikit-image/scikit-image',
          license='SciPy License (BSD Style)',
          **(configuration(top_path='').todict())
          )
