# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False

import os
import numpy as np
import pyopencl as cl

from skimage.util.clutil import roundUp, createProgram
from skimage._shared.Buffer2D import Buffer2D
from skimage._shared.StreamCompact import StreamCompact

cimport numpy as cnp
from libc.math cimport log

DEF szFloat = 4
DEF szInt = 4
DEF szChar = 1
cm = cl.mem_flags

DEF LEN_WORKGROUP = 256
cdef tuple LWORKGROUP_2D = (16, 16)

class Operator:
    EQUAL = 0
    GT = 1
    LT = 2
    GTE = 3
    LTE = 4

class Logical:
    AND = 0
    OR = 1

cdef class IncrementalTileList:
    cdef:
        readonly tuple dim
        readonly tuple global_dim
        tuple tiles_dim
        object context
        object queue
        tuple lw

        object kern_flag, kern_flag_logcal, kern_init,\
        kern_increment_logical, kern_filter

        object streamCompact

        cnp.ndarray h_length
        object d_length, d_flags
        readonly object d_list, d_tiles

        int is_dirty

        int init_iteration
        readonly int iteration

    def __init__(self, context, devices, global_dim, tiles_dim):
        self.h_length = np.empty((1,), np.int32)

        self.dim = (global_dim[0]/tiles_dim[0], global_dim[1]/tiles_dim[1])
        self.global_dim = global_dim
        self.tiles_dim = tiles_dim

        self.streamCompact = StreamCompact(context, devices,
            self.dim[0] * self.dim[1])

        self.d_length = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR,
            hostbuf=self.h_length)
        self.d_list = self.streamCompact.listFactory()

        self.queue = cl.CommandQueue(context)

        self.is_dirty = True

        self.d_tiles = Buffer2D.fromBuffer(self.streamCompact.flagFactory(),
            self.dim, np.int32)
        self.d_flags = Buffer2D.fromBuffer(self.streamCompact.flagFactory(),
            self.dim, np.int32)

        options = [
            '-D TILESW=' + str(self.dim[0]),
            '-D TILESH=' + str(self.dim[1]),
            '-D TILEW=' + str(tiles_dim[0]),
            '-D TILEH=' + str(tiles_dim[1])
        ]

        filename = os.path.join(os.path.dirname(__file__),
            'incremental_tile_list.cl')
        program = createProgram(context, devices, options, filename)

        self.kern_flag = cl.Kernel(program, 'flag')
        self.kern_flag_logcal = cl.Kernel(program, 'flag_logical')
        self.kern_init = cl.Kernel(program, 'init_incremental')
        self.kern_increment_logical = cl.Kernel(program, 'increment_logical')
        self.kern_filter = cl.Kernel(program, 'filter')

        self.init_iteration = -1

        self.lw = (LEN_WORKGROUP, )

        self.reset()

    @property
    def length(self):
        if self.is_dirty:
            cl.enqueue_copy(self.queue, self.h_length, self.d_length).wait()
            self.is_dirty = False

        return int(self.h_length[0])

    def increment(self):
        self.iteration += 1

        return self.iteration

    def build(self, operator=None, operand=None):
        if operator == None: operator = Operator.EQUAL
        if operand == None:  operand = self.iteration

        length = self.d_tiles.size / szInt

        gw = roundUp((length, ), self.lw)
        args = [
            self.d_tiles,
            self.d_flags,
            np.int32(length),
            np.int32(operator),
            np.int32(operand)
        ]
        self.kern_flag(self.queue, gw, self.lw, *args).wait()

        self.streamCompact.compact(self.d_flags, self.d_list, self.d_length)
        self.is_dirty = True

    def incorporate(self, d_tiles2, operator1, operand1, operator2, operand2,
                    logical):
        length = self.d_tiles.size / szInt

        gw = roundUp((length, ), self.lw)
        args = [
            self.d_tiles,
            d_tiles2,
            self.d_tiles,
            np.int32(length),
            np.int32(operator1),
            np.int32(operator2),
            np.int32(operand1),
            np.int32(operand2),
            np.int32(logical),
            np.int32(self.iteration)
        ]
        self.kern_increment_logical(self.queue, gw, self.lw, *args).wait()

    def buildLogical(self, d_tiles2, operator1, operand1, operator2, operand2,
                     logical):
        length = self.d_tiles.size / szInt

        gw = roundUp((length, ), IncrementalTileList.lw)
        args = [
            self.d_tiles,
            d_tiles2,
            self.d_flags,
            np.int32(length),
            np.int32(operator1),
            np.int32(operator2),
            np.int32(operand1),
            np.int32(operand2),
            np.int32(logical)
        ]
        self.kern_flag_logcal(self.queue, gw, self.lw, *args).wait()

        self.streamCompact.compact(self.d_flags, self.d_list, self.d_length)
        self.is_dirty = True

    def reset(self):
        self.iteration = self.init_iteration + 1

        args = [
            self.d_tiles,
            np.array(self.d_tiles.dim, np.int32),
            np.int32(self.init_iteration)
        ]

        gw = roundUp(self.dim, LWORKGROUP_2D)

        self.kern_init(self.queue, gw, LWORKGROUP_2D, *args).wait()

    def execute(self, queue, input):
        context = queue.get_info(cl.command_queue_info.CONTEXT)

        output = Buffer2D(context, cm.READ_WRITE, input.global_dim,
            dtype=np.uint32)

        args = [
            input.d_tiles,
            np.array(input.global_dim, np.int32),
            np.int32(input.iteration),
            output
        ]

        gWorksizeTiles16 = roundUp(input.global_dim, (16, 16))

        self.kern_filter(queue, gWorksizeTiles16, (16, 16), *args).wait()

        return output
