"""
Script to show the results of the two marching cubes algorithms on different
data.
"""

import time

import numpy as np
import visvis as vv

from skimage.measure import marching_cubes_classic, marching_cubes_lewiner
from skimage.draw import ellipsoid


# Create test volume
SELECT = 3
gradient_dir = 'descent'  # ascent or descent

if SELECT == 1:
    # Medical data
    vol = vv.volread('stent')
    isovalue = 800

elif SELECT == 2:
    # Blocky data
    vol = vv.aVolume(20, 128) # Different every time
    isovalue = 0.2

elif SELECT == 3:
    # Generate two donuts using a formula by Thomas Lewiner
    n = 48
    a, b = 2.5 / n, -1.25
    isovalue = 0.0
    #
    vol = np.empty((n, n, n), 'float32')
    for iz in range(vol.shape[0]):
        for iy in range(vol.shape[1]):
            for ix in range(vol.shape[2]):
                z, y, x = float(iz)*a+b, float(iy)*a+b, float(ix)*a+b
                vol[iz, iy, ix] = ( (
                    (8*x)**2 + (8*y-2)**2 + (8*z)**2 + 16 - 1.85*1.85 ) * ( (8*x)**2 +
                    (8*y-2)**2 + (8*z)**2 + 16 - 1.85*1.85 ) - 64 * ( (8*x)**2 + (8*y-2)**2 )
                    ) * ( ( (8*x)**2 + ((8*y-2)+4)*((8*y-2)+4) + (8*z)**2 + 16 - 1.85*1.85 )
                    * ( (8*x)**2 + ((8*y-2)+4)*((8*y-2)+4) + (8*z)**2 + 16 - 1.85*1.85 ) -
                    64 * ( ((8*y-2)+4)*((8*y-2)+4) + (8*z)**2
                    ) ) + 1025
    # Uncommenting the line below will yield different results for classic MC
    #vol = -vol

elif SELECT == 4:
    vol = ellipsoid(4, 3, 2, levelset=True)
    isovalue = 0

# Get surface meshes
t0 = time.time()
vertices1, faces1, *_ = marching_cubes_lewiner(vol, isovalue, gradient_direction=gradient_dir, use_classic=False)
print('finding surface lewiner took %1.0f ms' % (1000*(time.time()-t0)) )

t0 = time.time()
vertices2, faces2, *_ = marching_cubes_classic(vol, isovalue, gradient_direction=gradient_dir)
print('finding surface classic took %1.0f ms' % (1000*(time.time()-t0)) )

# Show
vv.figure(1); vv.clf()
a1 = vv.subplot(121); m1 = vv.mesh(np.fliplr(vertices1), faces1)
a2 = vv.subplot(122); m2 = vv.mesh(np.fliplr(vertices2), faces2)
a1.camera = a2.camera

# visvis uses right-hand rule, gradient_direction param uses left-hand rule
m1.cullFaces = m2.cullFaces = 'front'  # None, front or back

vv.use().Run()
