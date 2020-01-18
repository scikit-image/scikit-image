#!/usr/bin/env python

import os

from skimage._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('_shared', parent_package, top_path)
    config.add_data_dir('tests')

    cython(['_ccomp.pyx',
            'geometry.pyx',
            'transform.pyx',
            '_warps_cy.pyx',
            '_convex_hull.pyx',
            'interpolation.pyx',
            '_pnpoly.pyx'], working_path=base_path)

    config.add_extension('_ccomp', sources=['_ccomp.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('geometry', sources=['geometry.c'])
    config.add_extension('transform', sources=['transform.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_warps_cy', sources=['_warps_cy.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_convex_hull', sources=['_convex_hull.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('interpolation', sources=['interpolation.c'])
    config.add_extension('_pnpoly', sources=['_pnpoly.c'],
                         include_dirs=[get_numpy_include_dirs(), '../_shared'])
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='scikit-image Developers',
          author='scikit-image Developers',
          maintainer_email='scikit-image@python.org',
          description='Transforms',
          url='https://github.com/scikit-image/scikit-image',
          license='SciPy License (BSD Style)',
          **(configuration(top_path='').todict())
          )
