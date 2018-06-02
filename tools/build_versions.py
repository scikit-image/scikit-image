#!/usr/bin/env python

import numpy as np
import scipy as sp
import matplotlib as mpl
from PIL import Image
import Cython
import networkx


for m in (np, sp, mpl, Image, networkx, Cython):
    if m is Image:
        version = m.VERSION
    else:
        version = m.__version__
    print(m.__name__.rjust(10), ' ', version)
