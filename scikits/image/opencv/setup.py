#!/usr/bin/env python

import os
import shutil

current_dir = os.path.dirname(__file__)

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    config = Configuration('opencv', parent_package, top_path)
    
    config.add_data_dir(os.path.join(current_dir, 'tests'))
    
    # since distutils/cython has problems, we'll check to see if cython is 
    # installed and use that to rebuild the .c files, if not, we'll just build 
    # direct from the included .c files

    # if the cython compilation fails, it will overwrite the good .c files
    # so we back them up first
    
    src = os.path.join(current_dir, 'opencv_cv.c')
    dst = os.path.join(current_dir, 'opencv_cv.c.bak')
    shutil.copy(src, dst)
    src = os.path.join(current_dir, 'opencv_backend.c')
    dst = os.path.join(current_dir, 'opencv_backend.c.bak')
    shutil.copy(src, dst)    
    
    try:
        import cython  
        # no way to check the cython version number????
        os.system('cython opencv_backend.pyx')
        os.system('cython opencv_cv.pyx')
        
        # if the cython compilation failed, the resulting file 
        # size will be very small ~ 78 bytes
        f1 = os.path.join(current_dir, 'opencv_cv.c')
        f2 = os.path.join(current_dir, 'opencv_backend.c')
        size1 = os.path.getsize(f1)
        size2 = os.path.getsize(f2)
        assert size1 > 100 and size2 > 100
    except:
        print 'Cython compilation failed. Building from backups.'
        src = os.path.join(current_dir, 'opencv_cv.c.bak')
        dst = os.path.join(current_dir, 'opencv_cv.c')
        shutil.copy(src, dst)
        src = os.path.join(current_dir, 'opencv_backend.c.bak')
        dst = os.path.join(current_dir, 'opencv_backend.c')
        shutil.copy(src, dst)
        
    config.add_extension('opencv_backend', sources=['opencv_backend.c'],
                         include_dirs=[get_numpy_include_dirs()])
    
    config.add_extension('opencv_cv', sources=['opencv_cv.c'],
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
          
        
        
        
        