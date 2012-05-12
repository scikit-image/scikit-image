from distutils.core import setup
from distutils.extension import Extension
import numpy as np

ext_modules = [
    Extension('_punwrap2D',
              ['unwrap_phase.c', 'Munther_2D_unwrap.c'],
              include_dirs = [np.get_include(),],
              #libraries = ['unwrap2D'],
              ),
    ]

setup(
    name = 'punwrap',
    ext_modules = ext_modules,
    )
