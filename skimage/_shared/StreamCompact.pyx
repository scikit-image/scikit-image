#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: wraparound=False

import os
import numpy as np
import pyopencl as cl

from skimage.util.clutil import createProgram
from skimage.util.clutil cimport roundUp_tuple
from skimage._shared.PrefixSum import PrefixSum

from libc.math cimport log

DEF szFloat = 4
DEF szInt = 4
DEF szChar = 1
cm = cl.mem_flags

DEF LEN_WORKGROUP = 256

cdef class StreamCompact:
    cdef:
        int capacity
        object context
        object queue
        tuple lw

        object kernCompact
        object kernScan_subarrays

        object prefix_sum

    def __cinit__(self, context, devices, int capacity):
        self.capacity = capacity

    def __init__(self, context, devices, capacity):
        self.context = context
        self.queue = cl.CommandQueue(context)

        self.prefix_sum = PrefixSum(context, devices, capacity)

        filename = os.path.join(os.path.dirname(__file__), 'stream_compact.cl')
        program = createProgram(context, devices, [], filename)

        self.kernCompact = cl.Kernel(program, 'compact')

        self.lw = (LEN_WORKGROUP, )

    def flagFactory(self, length=None):
        if length == None:
            length = self.capacity
        elif length > self.capacity:
            raise ValueError('length > self.capacity: {0}, {1}'.format(length, self.capacity))

        return cl.Buffer(self.context, cl.mem_flags.READ_WRITE, length*szInt)

    def listFactory(self, length=None):
        if length == None:
            length = self.capacity
        elif length > self.capacity:
            raise ValueError('length > self.capacity: {0}, {1}'.format(length, self.capacity))

        return self.prefix_sum.factory(length)

    cpdef compact(self, dFlags, dList, dLength, length=None):
        if length == None:
            length = dFlags.size/szInt
        elif length > self.capacity:
            raise ValueError('length > self.capacity: {0}, {1}'.format(length, self.capacity))

        cl.enqueue_copy(self.queue, dList, dFlags).wait()
        self.prefix_sum.scan(dList, dLength, length)

        gw = roundUp_tuple((length, ), self.lw)

        self.kernCompact(self.queue, gw, self.lw,
            dFlags,
            dList,
            dList,
            np.int32(length)
        ).wait()
