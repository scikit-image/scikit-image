import numpy as np
import sys


def lookfor(what):
    """Do a keyword search on scikit-image docstrings.

    Parameters
    ----------
    what : str
        Words to look for.

    """
    return np.lookfor(what, sys.modules[__name__.split('.')[0]])
