#!/usr/bin/env python

import os
from skimage._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('c_inpaint', parent_package, top_path)

    cython(['cython_imp.pyx'], working_path=base_path)

    config.add_extension('_c_inpaint', sources=['cython_imp.c'])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='scikit-image developers',
          author='scikit-image developers',
          maintainer_email='scikit-image@googlegroups.com',
          description='Drawing',
          url='https://github.com/scikit-image/scikit-image',
          license='SciPy License (BSD Style)',
          **(configuration(top_path='').todict())
          )
