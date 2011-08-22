#!/usr/bin/env python

import os
import shutil
import hashlib

from scikits.image._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('transform', parent_package, top_path)
    config.add_data_dir('tests')

    cython(['_template.pyx'], working_path=base_path)

    config.add_extension('_template', sources=['_template.c'],
                         include_dirs=[get_numpy_include_dirs()])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer = 'Scikits.Image Developers',
          author = 'Scikits.Image Developers',
          maintainer_email = 'scikits-image@googlegroups.com',
          description = 'Transforms',
          url = 'http://stefanv.github.com/scikits.image/',
          license = 'SciPy License (BSD Style)',
          **(configuration(top_path='').todict())
          )
