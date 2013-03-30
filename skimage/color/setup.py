#!/usr/bin/env python


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('color', parent_package, top_path)
    config.add_data_files('colors.ini')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='scikit-image Developers',
          maintainer_email='scikit-image@googlegroups.com',
          description='Image Color Routines',
          url='https://github.com/scikit-image/scikit-image',
          license='Modified BSD',
          **(configuration(top_path='').todict())
          )
