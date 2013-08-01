import os
import numpy as np
import pyopencl as cl
from skimage._shared.PrefixSum import PrefixSum

szInt = 4

platforms = cl.get_platforms()

devices = platforms[0].get_devices()
devices = [devices[1]]
context = cl.Context(devices)
queue = cl.CommandQueue(context)

nSamples = 65536
prefixSum = PrefixSum(context, devices, nSamples)

hList = np.random.randint(0, 20, nSamples).astype(np.int32)
dList = prefixSum.factory()
cl.enqueue_copy(queue, dList, hList).wait()

hTotal = np.empty((1, ), np.int32)
dTotal = cl.Buffer(context, cl.mem_flags.READ_WRITE, 1*szInt)

prefixSum.scan(dList, dTotal, nSamples)
cl.enqueue_copy(queue, hTotal, dTotal).wait()

hTmp = np.empty((nSamples,), np.int32)
cl.enqueue_copy(queue, hTmp, dList).wait()

cl.enqueue_copy(queue, hTotal, dTotal).wait()
length = hTotal[0]

#check for correctness
assert(hList.sum() == hTotal)

print hTmp
print hList
print hTotal

#evaluate performance
import time
import csv
from evaluate import global_dims, iterations, tile_dims

columns = [
    'img dim',
    'mp',
    'tile dim',
    'num tiles',
    'total ms',
    'kernels ms'
]

print ', '.join(map(str, columns))

for global_dim in global_dims:
    for tile_dim in tile_dims:
        n_tiles = (global_dim[0]/tile_dim[0])*(global_dim[1]/tile_dim[1])
        prefixSum = PrefixSum(context, devices, n_tiles)

        dList = prefixSum.factory()
        dTotal = cl.Buffer(context, cl.mem_flags.READ_WRITE, 1*szInt)

        mp = float(global_dim[0]*global_dim[1])/(1024*1024)

        t = t2 = elapsed = 0
        t = time.time()
        for i in range(iterations):
            prefixSum.scan(dList, dTotal, n_tiles)

        t2 = time.time()
        elapsed += t2 - t

        row = [
            "({0}x{1})".format(global_dim[0], global_dim[1]),
            mp,
            "({0}x{1})".format(tile_dim[0], tile_dim[1]),
            n_tiles,
            (elapsed/iterations * 1000),
            (1e-6 * prefixSum.elapsed)/iterations
        ]

        print ', '.join(map(str, row))

True