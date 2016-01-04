#!/usr/bin/env python

from skimage._build import cython

import os
import tempfile
import shutil
from numpy.distutils.command.build_ext import build_ext
from distutils.errors import CompileError, LinkError

base_path = os.path.abspath(os.path.dirname(__file__))

compile_flags = ['-fopenmp']
link_flags = ['-fopenmp']

code = """#include <omp.h>
int main(int argc, char** argv) { return(0); }"""

class Checker(build_ext):

    def can_compile_link(self):

        cc = self.compiler
        fname = 'test.c'
        cwd = os.getcwd()
        tmpdir = tempfile.mkdtemp()

        try:
            os.chdir(tmpdir)
            with open(fname, 'wt') as fobj:
                fobj.write(code)
            try:
                objects = cc.compile([fname],
                                     extra_postargs=compile_flags)
            except CompileError:
                return False
            try:
                # Link shared lib rather then executable to avoid
                # http://bugs.python.org/issue4431 with MSVC 10+
                cc.link_shared_lib(objects, "testlib",
                                   extra_postargs=link_flags)
            except (LinkError, TypeError):
                return False
        finally:
            os.chdir(cwd)
            shutil.rmtree(tmpdir)
        return True

    def build_extensions(self):
        """ Hook into extension building to check compiler flags """

        if self.can_compile_link():

            for ext in self.extensions:
                ext.extra_compile_args += compile_flags
                ext.extra_link_args += link_flags

        build_ext.build_extensions(self)


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('detect', parent_package, top_path)
    config.add_data_dir('tests')

    # This function tries to create cpp files from the given .pyx files.  If
    # it fails, try to build with pre-generated .cpp files.


    cython(['cascade.pyx'], working_path=base_path)
    config.add_extension('cascade', sources=['cascade.cpp'],
                         include_dirs=[get_numpy_include_dirs()],
                         language="c++",
                         extra_compile_args=compile_flags,
                         extra_link_args=link_flags)
    return config

cmdclass = dict(build_ext=Checker)

if __name__ == '__main__':
    from numpy.distutils.core import setup

    conf = configuration(top_path='').todict()

    setup(maintainer='scikit-image Developers',
          maintainer_email='scikit-image@googlegroups.com',
          description='Object detection framework',
          url='https://github.com/scikit-image/scikit-image',
          license='Modified BSD',
          cmdclass=cmdclass,
          **(conf)
          )
