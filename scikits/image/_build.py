import sys
import os
import shutil
import hashlib
import subprocess
import platform

def cython(pyx_files, working_path=''):
    """Use Cython to convert the given files to C.

    Parameters
    ----------
    pyx_files : list of str
        The input .pyx files.

    """
    # Do not build cython files if target is clean
    if sys.argv[1] == 'clean':
        return

    try:
        import Cython
    except ImportError:
        # If cython is not found, we do nothing -- the build will make use of
        # the distributed .c files
        print("Cython not found; falling back to pre-built %s" \
              % " ".join([f.replace('.pyx', '.c') for f in pyx_files]))
    else:
        for pyxfile in [os.path.join(working_path, f) for f in pyx_files]:
            # make a backup of the good c files
            c_file = pyxfile[:-4] + '.c'
            c_file_new = c_file + '.new'

            # run cython compiler
            cmd = 'cython -o %s %s' % (c_file_new, pyxfile)
            print(cmd)

            if platform.system() == 'Windows':
                status = subprocess.call(
                    [sys.executable,
                     os.path.join(os.path.dirname(sys.executable),
                                  'Scripts', 'cython.py'),
                     '-o', c_file_new, pyxfile],
                    shell=True)
            else:
                status = subprocess.call(['cython', '-o', c_file_new, pyxfile])

            # if the resulting file is small, cython compilation failed
            if status != 0 or os.path.getsize(c_file_new) < 100:
                print("Cython compilation of %s failed. Falling back " \
                      "on pre-generated file." % os.path.basename(pyxfile))
            elif not same_cython(c_file_new, c_file):
                # if the generated .c file differs from the one provided,
                # use that one instead
                shutil.copy(c_file_new, c_file)
            os.remove(c_file_new)


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

    f0 = open(f0, 'rb')
    f0.readline()

    f1 = open(f1, 'rb')
    f1.readline()

    return md5sum(f0) == md5sum(f1)

