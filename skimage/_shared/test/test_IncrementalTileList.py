import numpy as np
import pyopencl as cl
from skimage._shared.IncrementalTileList import IncrementalTileList, Operator

szFloat =  4
szInt = 4
szChar = 1
cm = cl.mem_flags

platforms = cl.get_platforms()

devices = platforms[0].get_devices()
devices = [devices[1]]
context = cl.Context(devices)
queue = cl.CommandQueue(context)

global_dim = (4096, 4096)
global_shape = (global_dim[1], global_dim[0])

tileList = IncrementalTileList(context, devices, global_dim, (16, 16))
dim = tileList.dim

hTiles = np.random.randint(0, 20, (dim[1], dim[0])).astype(np.int32)
cl.enqueue_copy(queue, tileList.d_tiles, hTiles).wait()

tileList.build(Operator.GTE, 10)

hList = np.empty((dim[0]*dim[1],), np.int32)
cl.enqueue_copy(queue, hList, tileList.d_list).wait()

print hTiles
print 'dim: {0}, num elements: {1}'.format(dim, dim[0]*dim[1])

print hList

#test correctness
compact_cpu = np.where(hTiles >= 10)
compact_cpu = map(lambda x, y: y*dim[0] + x, compact_cpu[1], compact_cpu[0])
assert(np.all(compact_cpu == hList[0:tileList.length]))

#evaluate performance
import time
from evaluate import global_dims, iterations, tile_dims, columns

print ', '.join(map(str, columns))

for global_dim in global_dims:
    for tile_dim in tile_dims:
        tileList = IncrementalTileList(context, devices, global_dim, tile_dim)

        mp = float(global_dim[0]*global_dim[1])/(1024*1024)

        t = elapsed = 0
        for i in xrange(iterations):
            t = time.time()

            tileList.build(Operator.GTE, 10)

            elapsed += time.time()-t

        row = [
                "({0} {1})".format(global_dim[0], global_dim[1]),
                mp,
                "({0} {1})".format(tile_dim[0], tile_dim[1]),
                tileList.dim[0]*tileList.dim[1],
                (elapsed/iterations * 1000)
            ]

        print ', '.join(map(str, row))

True