import numpy as np

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("crank8", ["_crank8.pyx"], include_dirs=[np.get_include()]),
                    Extension("crank8_percentiles", ["_crank8_percentiles.pyx"], include_dirs=[np.get_include()]),
                    Extension("crank16", ["_crank16.pyx"], include_dirs=[np.get_include()]),
                    Extension("crank16_bilateral", ["_crank16_bilateral.pyx"], include_dirs=[np.get_include()]),
                    Extension("crank16_percentiles", ["_crank16_percentiles.pyx"], include_dirs=[np.get_include()])]
)


##!/usr/bin/env python
#
#import os
#from skimage._build import cython
#
#base_path = os.path.abspath(os.path.dirname(__file__))
#
#
#def configuration(parent_package='', top_path=None):
#    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
#
#    config = Configuration('rank', parent_package, top_path)
#    config.add_data_dir('tests')
#
#    cython(['_texture.pyx'], working_path=base_path)
#    cython(['_template.pyx'], working_path=base_path)
#
#    config.add_extension('_texture', sources=['_texture.c'],
#        include_dirs=[get_numpy_include_dirs(), '../_shared'])
#    config.add_extension('_template', sources=['_template.c'],
#        include_dirs=[get_numpy_include_dirs(), '../_shared'])
#
#    return config
#
#if __name__ == '__main__':
#    from numpy.distutils.core import setup
#    setup(maintainer='scikits-image Developers',
#        author='scikits-image Developers',
#        maintainer_email='scikits-image@googlegroups.com',
#        description='Features',
#        url='https://github.com/scikits-image/scikits-image',
#        license='SciPy License (BSD Style)',
#        **(configuration(top_path='').todict())
#    )
