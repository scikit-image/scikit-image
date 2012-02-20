#!/usr/bin/env python

import os
from skimage._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('resample', parent_package, top_path)
    config.add_data_dir('tests')

    cython(['_resample.pyx'], working_path=base_path)

    config.add_extension(
        '_resample',
        sources=['_resample.c'],
        include_dirs=[get_numpy_include_dirs()],
        extra_compile_args=[
            "-fopenmp",
            "-O3",
            "-mtune=native",
            "-funroll-loops",
        ],
        extra_link_args=['-fopenmp'],
        )

    return config
