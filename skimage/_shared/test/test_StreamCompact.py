import os
import numpy as np
import pyopencl as cl
from skimage._shared.StreamCompact import StreamCompact

szFloat =  4
szInt = 4
szChar = 1
cm = cl.mem_flags

platforms = cl.get_platforms()

devices = platforms[0].get_devices()
devices = [devices[1]]
context = cl.Context(devices)
queue = cl.CommandQueue(context)

nSamples = 65536
capcity = nSamples

streamCompact = StreamCompact(context, devices, capcity)

hList = np.empty((nSamples,), np.int32)
dList = streamCompact.listFactory(nSamples)

hFlags = np.random.randint(0, 2, nSamples).astype(np.int32)
dFlags = streamCompact.flagFactory(nSamples)
cl.enqueue_copy(queue, dFlags, hFlags).wait()

hLength = np.empty((1, ), np.int32)
dLength = cl.Buffer(context, cl.mem_flags.READ_WRITE, 1*szInt)

streamCompact.compact(dFlags, dList, dLength, nSamples)
cl.enqueue_copy(queue, hList, dList).wait()
cl.enqueue_copy(queue, hLength, dLength).wait()

print 'flags', hFlags

#test correctness
compact_cpu = np.where(hFlags == 1)[0]
assert(np.all(compact_cpu == hList[0:hLength]))

#evaluate performance
import time
from evaluate import global_dims, iterations, tile_dims, columns

print ', '.join(map(str, columns))

for global_dim in global_dims:
    for tile_dim in tile_dims:
        n_tiles = (global_dim[0]/tile_dim[0])*(global_dim[1]/tile_dim[1])
        streamCompact = StreamCompact(context, devices, n_tiles)

        dList = streamCompact.listFactory()
        dFlags = streamCompact.flagFactory()
        dLength = cl.Buffer(context, cl.mem_flags.READ_WRITE, 1*szInt)

        mp = float(global_dim[0]*global_dim[1])/(1024*1024)

        t = elapsed = 0
        for i in xrange(iterations):
            cl.enqueue_copy(queue, dFlags, dFlags).wait()

            t = time.time()

            streamCompact.compact(dFlags, dList, dLength, n_tiles)

            elapsed += time.time()-t

        row = [
            "({0} {1})".format(global_dim[0], global_dim[1]),
            mp,
            "({0} {1})".format(tile_dim[0], tile_dim[1]),
            n_tiles,
            (elapsed/iterations * 1000)
        ]

        print ', '.join(map(str, row))

True