"""Image Processing SciKit (Toolbox for SciPy)"""


import os.path as _osp

pkg_dir = _osp.abspath(_osp.dirname(__file__))
data_dir = _osp.join(pkg_dir, 'data')

from version import version as __version__

def _setup_test():
    import gzip
    import functools

    basedir = _osp.join(pkg_dir, '../')
    args = ['', '--exe', '-w', '%s' % basedir]

    try:
        import nose as _nose
    except ImportError:
        print("Could not load nose.  Unit tests not available.")
        return None
    else:
        return functools.partial(_nose.run, 'scikits.image', argv=args)

test = _setup_test()
if test is None:
    del test

def get_log(name):
    """Return a console logger.

    Output may be sent to the logger using the `debug`, `info`, `warning`,
    `error` and `critical` methods.

    Parameters
    ----------
    name : str
        Name of the log.

    References
    ----------
    .. [1] Logging facility for Python,
           http://docs.python.org/library/logging.html

    """
    import logging, sys
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
    return logging.getLogger(name)

from .util.dtype import *
