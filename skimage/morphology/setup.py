#!/usr/bin/env python

import os
from skimage._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('morphology', parent_package, top_path)
    config.add_data_dir('tests')

    cython(['_watershed.pyx'], working_path=base_path)
    cython(['_skeletonize_cy.pyx'], working_path=base_path)
    cython(['_convex_hull.pyx'], working_path=base_path)
    cython(['_greyreconstruct.pyx'], working_path=base_path)
    cython(['_skeletonize_3d_cy.pyx.in'], working_path=base_path)
    cython(['_criteria.pyx'], working_path=base_path)
    
    config.add_extension('_watershed', sources=['_watershed.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_skeletonize_cy', sources=['_skeletonize_cy.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_convex_hull', sources=['_convex_hull.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_greyreconstruct', sources=['_greyreconstruct.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_skeletonize_3d_cy', sources=['_skeletonize_3d_cy.c'],
                         include_dirs=[get_numpy_include_dirs()])
#     config.add_extension("criteria_classes",
#                          sources=["criteria_classes.pyx", "criteria_classes_inc.cpp"], 
#                          include_dirs=[get_numpy_include_dirs()],
#                          language="c++")
#    config.add_extension('_criteria', sources=['_criteria.c'],
#                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_criteria', 
                         sources=['_criteria.c', 'criteria_classes.pyx', 'criteria_classes_inc.cpp'],
                         language='c++',
                         include_dirs=[get_numpy_include_dirs()])

    return config

#from distutils.core import setup
#from distutils.extension import Extension
#from Cython.Distutils import build_ext

#ext = Extension("wrap_area", ["wrap_area.pyx", "area.cpp"], language="c++")

#setup(
#    cmdclass = {'build_ext': build_ext},
#    ext_modules = [ext],
#)

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='scikit-image Developers',
          author='Damian Eads',
          maintainer_email='scikit-image@python.org',
          description='Morphology Wrapper',
          url='https://github.com/scikit-image/scikit-image',
          license='SciPy License (BSD Style)',
          **(configuration(top_path='').todict())
          )
