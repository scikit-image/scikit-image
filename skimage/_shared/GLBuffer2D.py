import pyopencl as cl
from skimage.util.clutil import alignedDim, padArray2D
import numpy as np

LWORKGROUP = (16, 16)

class GLBuffer2D(cl.Buffer):
    def __init__(self, context, flags, dim=None, dtype=None, hostbuf=None):
        if hostbuf != None:
            dim = (hostbuf.shape[1], hostbuf.shape[0])
            dtype = hostbuf.dtype
        else:
            if dim == None or dtype == None:
                raise ValueError('Dimensions and datatype required')

        pad = alignedDim(dim, dtype)

        if hostbuf != None:
            hostbuf = padArray2D(hostbuf, (hostbuf.shape[0], pad), 'constant')
            cl.Buffer.__init__(self, context, flags, hostbuf=hostbuf)
        else:
            cl.Buffer.__init__(self, context, flags,
                size=np.dtype(dtype).itemsize*pad*dim[1])

        self.dim = dim
        self.dtype = dtype

    @staticmethod
    def fromBuffer(buffer, dim, dtype):
        buffer.dim = dim
        buffer.dtype = dtype

        return buffer
