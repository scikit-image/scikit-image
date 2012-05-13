from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
#from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension('unwrap2D', 
              ['unwrap2D.pyx',
               'Miguel_2D_unwrapper_with_mask_and_wrap_around_option.c',
               ],
              include_dirs = [np.get_include(),],
              ),
    Extension('unwrap3D', 
              ['unwrap3D.pyx',
               'Hussein_3D_unwrapper_with_mask_and_wrap_around_option.c',
               ],
              include_dirs = [np.get_include(),],
              ),
    ]

import numpy as np

setup(
    name = 'unwrap',
    #ext_modules = cythonize(['cytransient.pyx',], 
    #                        include_path = [np.get_include(),],
    #                        ),
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    )
