import sys
import os
import hashlib
import subprocess


def cython(pyx_files, working_path=''):
    """Use Cython to convert the given files to C.

    Parameters
    ----------
    pyx_files : list of str
        The input .pyx files.

    """
    # Do not build cython files if target is clean
    if len(sys.argv) >= 2 and sys.argv[1] == 'clean':
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

            # if the .pyx file stayed the same, we don't need to recompile
            if not _changed(pyxfile):
                continue

            c_file = pyxfile[:-4] + '.c'

            # run cython compiler
            cmd = 'cython -o %s %s' % (c_file, pyxfile)
            print(cmd)

            try:
                subprocess.call(['cython', '-o', c_file, pyxfile])
            except WindowsError:
                # On Windows cython.exe may be missing if Cython was installed
                # via distutils. Run the cython.py script instead.
                subprocess.call(
                    [sys.executable,
                     os.path.join(os.path.dirname(sys.executable),
                                  'Scripts', 'cython.py'),
                     '-o', c_file, pyxfile],
                    shell=True)


def _md5sum(f):
    m = hashlib.new('md5')
    while True:
        # Hash one 8096 byte block at a time
        d = f.read(8096)
        if not d:
            break
        m.update(d)
    return m.hexdigest()


def _changed(filename):
    """Compare the hash of a Cython file to the cached hash value on disk.

    """
    filename_cache = filename + '.md5'

    try:
        md5_cached = open(filename_cache, 'rb').read()
    except IOError:
        md5_cached = '0'

    with open(filename, 'rb') as f:
        md5_new = _md5sum(f)

        with open(filename_cache, 'wb') as cf:
            cf.write(md5_new.encode('utf-8'))

    return md5_cached != md5_new
