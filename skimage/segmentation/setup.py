#!/usr/bin/env python

import os
from skimage._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('segmentation', parent_package, top_path)

    cython(['_watershed_cy.pyx',
            '_felzenszwalb_cy.pyx',
            '_quickshift_cy.pyx',
            '_slic.pyx',
            '_remap.pyx'], working_path=base_path)
    config.add_extension('_watershed_cy', sources=['_watershed_cy.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_felzenszwalb_cy', sources=['_felzenszwalb_cy.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_quickshift_cy', sources=['_quickshift_cy.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_slic', sources=['_slic.c'],
                         include_dirs=[get_numpy_include_dirs()])
    # note: the extra compiler flag -std=c++0x is needed to access the
    # std::unordered_map container on some earlier gcc compilers. See:
    # https://stackoverflow.com/a/3973692/224254
    config.add_extension('_remap', sources='_remap.cpp',
                         include_dirs=[get_numpy_include_dirs()],
                         language='c++', extra_compile_args=['-std=c++0x'])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='scikit-image Developers',
          maintainer_email='scikit-image@python.org',
          description='Segmentation Algorithms',
          url='https://github.com/scikit-image/scikit-image',
          license='SciPy License (BSD Style)',
          **(configuration(top_path='').todict())
          )
