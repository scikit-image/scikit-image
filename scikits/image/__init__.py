"""Image Processing SciKit (Toolbox for SciPy)"""


import os.path as _osp

data_dir = _osp.join(_osp.dirname(__file__), 'data')

from version import version as __version__

def _setup_test():
    import functools

    basedir = _osp.dirname(_osp.join(__file__, '../'))
    args = ['', '--exe', '-w', '%s' % basedir]

    try:
        import nose as _nose
    except ImportError:
        print "Could not load nose.  Unit tests not available."
        return None
    else:
        return functools.partial(_nose.run, 'scikits.image', argv=args)

test = _setup_test()
if test is None:
    del test

