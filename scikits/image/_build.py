import os
import shutil
import hashlib

def cython(pyx_files, working_path=''):
    """Use Cython to convert the given files to C.

    Parameters
    ----------
    pyx_files : list of str
        The input .pyx files.

    """
    try:
        import Cython
    except ImportError:
        # If cython is not found, we do nothing -- the build will make use of
        # the distributed .c files
        pass
    else:
        for pyxfile in [os.path.join(working_path, f) for f in pyx_files]:
            # make a backup of the good c files
            c_file = pyxfile.rstrip('pyx') + 'c'
            c_file_new = c_file + '.new'

            # run cython compiler
            cmd = 'cython -o %s %s' % (c_file_new, pyxfile)
            print cmd
            status = os.system(cmd)

            # if the resulting file is small, cython compilation failed
            size = os.path.getsize(c_file_new)
            if status != 0 or (size < 100):
                print "Cython compilation of %s failed. Falling back " \
                      "on pre-generated file." % os.path.basename(pyxfile)
                continue

            # if the generated .c file differs from the one provided,
            # use that one instead
            if not same_cython(c_file_new, c_file):
                shutil.copy(c_file_new, c_file)

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

    if not (os.path.isfile(f0) and os.path.isfile(f1)):
        return False

    f0 = file(f0)
    f0.readline()

    f1 = file(f1)
    f1.readline()

    return md5sum(f0) == md5sum(f1)
