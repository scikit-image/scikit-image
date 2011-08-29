#!/usr/bin/env python

from scikits.image._build import cython

import os.path

base_path = os.path.abspath(os.path.dirname(__file__))

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('opencv', parent_package, top_path)

    config.add_data_dir('tests')

    cython_files = ['opencv_backend.pyx', 'opencv_cv.pyx']

    # This function tries to create C files from the given .pyx files.  If
    # it fails, we build the checked-in .c files.
    cython(cython_files, working_path=base_path)

    for pyxfile in cython_files:
        c_file = pyxfile[:-4] + '.c'
        config.add_extension(pyxfile[:-4],
                             sources=[c_file],
                             include_dirs=[get_numpy_include_dirs()])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer = 'Scikits.Image Developers',
          author = 'Steven C. Colbert',
          maintainer_email = 'scikits-image@googlegroups.com',
          description = 'OpenCV wrapper for NumPy arrays',
          url = 'https://github.com/scikits-image/scikits.image',
          license = 'SciPy License (BSD Style)',
          **(configuration(top_path='').todict())
          )
