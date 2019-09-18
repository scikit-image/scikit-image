"""


mweigert@mpi-cbg.de
"""

from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
from .quaternion import Quaternion

def mat4_rotate(w=0,x=1,y=0,z=0):
    n = np.array([x,y,z]).astype(np.float32)
    n *= 1./np.sqrt(1.*np.sum(n**2))
    q = Quaternion(np.cos(.5*w),*(np.sin(.5*w)*n))
    return q.toRotation4()


def mat4_translate(x=0,y=0,z=0):
    M = np.identity(4)
    M[:3,3] = x,y,z
    return M

def mat4_scale(sx=1.0, sy =1.0, sz = 1.):
    M = np.identity(4)
    M[0,0] = sx
    M[1, 1] = sy
    M[2 ,2] = sz
    return M
