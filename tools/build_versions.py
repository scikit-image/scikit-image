#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import scipy as sp
import matplotlib as mpl
import six
from PIL import Image
import Cython
import networkx


for m in (np, sp, mpl, six, Image, networkx, Cython):
    if m is Image:
        version = m.VERSION
    else:
        version = m.__version__
    print(m.__name__.rjust(10), ' ', version)
