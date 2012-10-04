import numpy as np

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("cmorph", ["cmorph.pyx"], include_dirs=[np.get_include()]),
                    Extension("crank", ["crank.pyx"], include_dirs=[np.get_include()]),
                    Extension("crank_percentiles", ["crank_percentiles.pyx"], include_dirs=[np.get_include()]),
                    Extension("crank16", ["crank16.pyx"], include_dirs=[np.get_include()]),
                    Extension("crank16_bilateral", ["crank16_bilateral.pyx"], include_dirs=[np.get_include()]),
                    Extension("crank16_percentiles", ["crank16_percentiles.pyx"], include_dirs=[np.get_include()])]
)



