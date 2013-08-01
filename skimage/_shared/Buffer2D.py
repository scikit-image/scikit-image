import pyopencl as cl
from skimage.util.clutil import alignedDim, padArray2D
import numpy as np

LWORKGROUP = (16, 16)

class Buffer2D(cl.Buffer):
    def __init__(self, context, flags, dim=None, dtype=None, hostbuf=None,
                 pad=True):
        if hostbuf != None:
            dim = (hostbuf.shape[1], hostbuf.shape[0])
            dtype = hostbuf.dtype
        else:
            if dim == None or dtype == None:
                raise ValueError('Dimensions and datatype required')

        if pad:
            pad = alignedDim(dim, dtype)

        if hostbuf != None:
            if pad:
                hostbuf = padArray2D(hostbuf, (hostbuf.shape[0], pad), 'constant')

            cl.Buffer.__init__(self, context, flags | cl.mem_flags.COPY_HOST_PTR,
                hostbuf=hostbuf)
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
