#!/usr/bin/env python

import os
import shutil
import hashlib

base_path = os.path.dirname(__file__)

def same_cython(f0, f1):
    '''Compare two Cython generated C-files, based on their md5-sum.

    Returns True if the files are identical, False if not.  The first
    lines are skipped, due to the timestamp printed there.

    '''
    def md5sum(f):
        m = hashlib.new('md5')
        while True:
            d = f.read(8096)
            if not d:
                break
            m.update(d)
        return m.hexdigest()

    f0 = file(f0)
    f0.readline()

    f1 = file(f1)
    f1.readline()

    return md5sum(f0) == md5sum(f1)


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
            c_file = pyxfile[:-4] + 'c'
            c_file_new = c_file + '.new'

            # run cython compiler
            os.system('cython -o %s %s' % (c_file_new, pyxfile))

            # if the resulting file is small, cython compilation failed
            size = os.path.getsize(c_file_new)
            if size < 100:
                print "Cython compilation of %s failed. Using " \
                      "pre-generated file." % os.path.basename(pyxfile)
                continue

            # if the generated .c file differs from the one provided,
            # use that one instead
            if not same_cython(c_file_new, c_file):
                shutil.copy(c_file_new, c_file)

    except ImportError:
        # if cython is not found, we just build from the included .c files
        pass

    for pyxfile in cython_files:
        c_file = pyxfile[:-4] + 'c'
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
          url = 'http://stefanv.github.com/scikits.image/',
          license = 'SciPy License (BSD Style)',
          **(configuration(top_path='').todict())
          )
