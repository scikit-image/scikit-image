#!/usr/bin/env python

import os
import shutil

base_path = os.path.dirname(__file__)

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('opencv', parent_package, top_path)

    config.add_data_dir('tests')

    # since distutils/cython has problems, we'll check to see if cython is
    # installed and use that to rebuild the .c files, if not, we'll just build
    # directly from the included .c files

    cython_files = ['opencv_backend.pyx', 'opencv_cv.pyx']

    try:
        import Cython
        for pyxfile in [os.path.join(base_path, f) for f in cython_files]:
            # make a backup of the good c files
            c_file = pyxfile.rstrip('pyx') + 'c'
            src = c_file
            dst = c_file + '.bak'
            shutil.copy(src, dst)

            # run cython compiler
            os.system('cython ' + pyxfile)

            # if the file is small, cython compilation failed
            size = os.path.getsize(c_file)
            if size < 100:
                print 'Cython compilation failed. Restoring from backup.'
                # restore from backup
                shutil.copy(dst, src)

    except ImportError:
        # if cython is not found, we just build from the include .c files
        pass

    for pyxfile in cython_files:
        c_file = pyxfile.rstrip('pyx') + 'c'
        config.add_extension(pyxfile.rstrip('.pyx'),
                             sources=[c_file],
                             include_dirs=[get_numpy_include_dirs()])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer = 'Scikits.Image Developers',
          author = 'Steven C. Colbert',
          maintainer_email = 'scikits-image@googlegroups.com',
          description = 'OpenCV wrapper for NumPy arrays',
          url = 'http://stefanv.github.com/scikits.image/',
          license = 'SciPy License (BSD Style)',
          **(configuration(top_path='').todict())
          )
