import pyopencl as cl

LWORKGROUP = (16, 16)

class Image2D(cl.Image):
    def __init__(self, context, flags, format, dim):
        cl.Image.__init__(self, context, flags, format, dim)

        self.dim = dim
