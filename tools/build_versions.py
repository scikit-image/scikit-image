#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import scipy as sp
from PIL import Image
import six

for m in (np, sp, Image, six):
    if not m is None:
        if m is Image:
            print('PIL'.rjust(10), ' ', Image.VERSION)
        else:
            print(m.__name__.rjust(10), ' ', m.__version__)
