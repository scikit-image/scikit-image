# Author: Virgile Fritsch <virgile.fritsch@inria.fr>
# License: BSD 3 clause

import numpy
from skimage._build import cython

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('__check_build', parent_package, top_path)

    cython(['_check_build.pyx'], working_path=base_path)
    config.add_extension('_check_build',
                         sources=['_check_build.c'])
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
