#!/usr/bin/env python

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('find_contours', parent_package, top_path)
    config.add_data_dir('tests')


    config.add_extension('_find_contours', sources=['_find_contours.c'],
                         include_dirs=[get_numpy_include_dirs()])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer = 'scikits.image Developers',
          maintainer_email = 'scikits-image@googlegroups.com',
          description = 'Graph-based Image-processing Algorithms',
          url = 'https://github.com/scikits-image/scikits.image',
          license = 'Modified BSD',
          **(configuration(top_path='').todict())
          )
