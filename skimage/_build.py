import sys
import os
import hashlib
import subprocess

# WindowsError is not defined on unix systems
try:
    WindowsError
except NameError:
    class WindowsError(Exception):
        pass


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
        from Cython.Build import cythonize
    except ImportError:
        # If cython is not found, we do nothing -- the build will make use of
        # the distributed .c files
        print("Cython not found; falling back to pre-built %s" \
              % " ".join([f.replace('.pyx.in', 'c').replace('.pyx', '.c')
                          for f in pyx_files]))
    else:
        for pyxfile in [os.path.join(working_path, f) for f in pyx_files]:

            # if the .pyx file stayed the same, we don't need to recompile
            if not _changed(pyxfile):
                continue

            if pyxfile.endswith('.pyx.in'):
                process_tempita_pyx(pyxfile)
                pyxfile = pyxfile.replace('.pyx.in', '.pyx')

            cythonize(pyxfile)

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

    return md5_cached != md5_new.encode('utf-8')


def process_tempita_pyx(fromfile):
    try:
        try:
            from Cython import Tempita as tempita
        except ImportError:
            import tempita
    except ImportError:
        raise Exception('Building requires Tempita: '
                        'pip install --user Tempita')
    from_filename = tempita.Template.from_filename
    template = from_filename(fromfile) #, encoding=sys.getdefaultencoding())
    pyxcontent = template.substitute()
    assert fromfile.endswith('.pyx.in')
    pyxfile = fromfile[:-len('.pyx.in')] + '.pyx'
    with open(pyxfile, "w") as f:
        f.write(pyxcontent)
