__author__ = 'marcdeklerk'

import os
from skimage._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('dtw', parent_package, top_path)

    cython(['_dtw.pyx'], working_path=base_path)

    config.add_extension('_dtw', sources=['_dtw.c'],
                         include_dirs=[get_numpy_include_dirs()])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='scikit-image Developers',
          maintainer_email='scikit-image@googlegroups.com',
          description='Cython Dynamic Time Warpings',
          url='https://github.com/scikit-image/scikit-image',
          license='SciPy License (BSD Style)',
          **(configuration(top_path='').todict())
          )