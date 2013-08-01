import os
import numpy as np
import pyopencl as cl

from skimage.util.clutil import roundUp, createProgram
from skimage._shared.Buffer2D import Buffer2D
from skimage._shared.Image2D import Image2D

LWORKGROUP = (16, 16)

class Colorize():
    class HUES():
        STANDARD = (0, 240.0 / 360)
        REVERSED = (240.0 / 360, 0)

    class SATURATION():
        STANDARD = 0

    def __init__(self, context, range, hues=None, sats=None, vals=None):
        self.context = context

        devices = context.get_info(cl.context_info.DEVICES)

        filename = os.path.join(os.path.dirname(__file__), 'colorize.cl')
        program = createProgram(context, devices, [], filename)

        self.kern_ui8 = cl.Kernel(program, 'colorize_ui8')
        self.kern_ui = cl.Kernel(program, 'colorize_ui32')
        self.kern_i = cl.Kernel(program, 'colorize_i32')
        self.kern_f = cl.Kernel(program, 'colorize_f32')

        self.format = format
        self.range = range
        self.hues = hues if hues != None else Colorize.HUES.STANDARD
        self.vals = vals if vals != None else (1, 1)
        self.sats = sats if sats != None else (1, 1)

    def execute(self, queue, input):
        output = Image2D(self.context,
            cl.mem_flags.READ_WRITE, cl.ImageFormat(cl.channel_order.RGBA,
                cl.channel_type.UNORM_INT8), input.dim)

        gw = roundUp(input.dim, LWORKGROUP)

        args = [
            np.array(self.range, np.float32),
            np.array(self.hues, np.float32),
            np.array(self.vals, np.float32),
            np.array(self.sats, np.float32),
            input,
            output,
            np.array(input.dim, np.int32)
        ]

        if type(input) == Buffer2D:
            if input.dtype == np.uint8:
                self.kern_ui8(queue, gw, LWORKGROUP, *args).wait()
            elif input.dtype == np.uint32:
                self.kern_ui(queue, gw, LWORKGROUP, *args).wait()
            elif input.dtype == np.int32:
                self.kern_i(queue, gw, LWORKGROUP, *args).wait()
            elif input.dtype == np.float32:
                self.kern_f(queue, gw, LWORKGROUP, *args).wait()
            else:
                raise NotImplementedError

        return output
