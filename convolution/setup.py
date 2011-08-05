
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy, os
os.system("rm ext.so")
setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("ext", ["ext.pyx"], extra_compile_args=["-ffast-math"])],
    include_dirs = [numpy.get_include(),],
)
