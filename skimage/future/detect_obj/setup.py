#!/usr/bin/env python

from skimage._build import cython
import os.path

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('detect_obj', parent_package, top_path)
    config.add_data_dir('tests')

    # This function tries to create C files from the given .pyx files.  If
    # it fails, try to build with pre-generated .c files.
    cython(['cascade.pyx'], working_path=base_path)
    config.add_extension('cascade', sources=['cascade.c'],
                         include_dirs=[get_numpy_include_dirs()],
                         extra_compile_args=['-fopenmp'],
                         extra_link_args=['-fopenmp'])
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='scikit-image Developers',
          maintainer_email='scikit-image@googlegroups.com',
          description='Object detection framework',
          url='https://github.com/scikit-image/scikit-image',
          license='Modified BSD',
          **(configuration(top_path='').todict())
          )
