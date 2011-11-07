"""Image Processing SciKit (Toolbox for SciPy)"""


import os.path as _osp

pkg_dir = _osp.abspath(_osp.dirname(__file__))
data_dir = _osp.join(pkg_dir, 'data')

from version import version as __version__

def _setup_test(verbose=False):
    import gzip
    import functools

    args = ['', '--exe', '-w', pkg_dir]
    if verbose:
        args.extend(['-v', '-s'])

    try:
        import nose as _nose
    except ImportError:
        print("Could not load nose.  Unit tests not available.")
        return None
    else:
        f = functools.partial(_nose.run, 'skimage', argv=args)
        f.__doc__ = 'Invoke the skimage test suite.'
        return f

test = _setup_test()
if test is None:
    del test

test_verbose = _setup_test(verbose=True)
if test_verbose is None:
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
