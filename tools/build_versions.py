#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import scipy as sp
import matplotlib as mpl
import six

for m in (np, sp, mpl, six):
    if not m is None:
        print(m.__name__.rjust(10), ' ', m.__version__)
