#!/usr/bin/env python

import os
from skimage._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('morphology', parent_package, top_path)
    config.add_data_dir('tests')

    cython(['ccomp.pyx'], working_path=base_path)
    cython(['cmorph.pyx'], working_path=base_path)
    cython(['_watershed.pyx'], working_path=base_path)
    cython(['_skeletonize.pyx'], working_path=base_path)
    cython(['_pnpoly.pyx'], working_path=base_path)
    cython(['_convex_hull.pyx'], working_path=base_path)

    config.add_extension('ccomp', sources=['ccomp.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('cmorph', sources=['cmorph.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_watershed', sources=['_watershed.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_skeletonize', sources=['_skeletonize.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_pnpoly', sources=['_pnpoly.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_convex_hull', sources=['_convex_hull.c'],
                         include_dirs=[get_numpy_include_dirs()])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer = 'scikits-image Developers',
          author = 'Damian Eads',
          maintainer_email = 'scikits-image@googlegroups.com',
          description = 'Morphology Wrapper',
          url = 'https://github.com/scikits-image/scikits-image',
          license = 'SciPy License (BSD Style)',
          **(configuration(top_path='').todict())
          )
