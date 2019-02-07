"""
Script to show the results of the two marching cubes algorithms on different
data.
"""

from __future__ import print_function

import time

import numpy as np

from skimage.measure import marching_cubes_classic, marching_cubes_lewiner
from skimage.draw import ellipsoid


def main(select=3, **kwargs):
    """Script main function.

    select: int
        1: Medical data
        2: Blocky data, different every time
        3: Two donuts
        4: Ellipsoid

    """
    import visvis as vv  # noqa: delay import visvis and GUI libraries

    # Create test volume
    if select == 1:
        vol = vv.volread('stent')
        isovalue = kwargs.pop('level', 800)
    elif select == 2:
        vol = vv.aVolume(20, 128)
        isovalue = kwargs.pop('level', 0.2)
    elif select == 3:
        with Timer('computing donuts'):
            vol = donuts()
        isovalue = kwargs.pop('level', 0.0)
        # Uncommenting the line below will yield different results for
        # classic MC
        # vol *= -1
    elif select == 4:
        vol = ellipsoid(4, 3, 2, levelset=True)
        isovalue = kwargs.pop('level', 0.0)
    else:
        raise ValueError('invalid selection')

    # Get surface meshes
    with Timer('finding surface lewiner'):
        vertices1, faces1 = marching_cubes_lewiner(vol, isovalue, **kwargs)[:2]

    with Timer('finding surface classic'):
        vertices2, faces2 = marching_cubes_classic(vol, isovalue, **kwargs)

    # Show
    vv.figure(1)
    vv.clf()
    a1 = vv.subplot(121)
    vv.title('Lewiner')
    m1 = vv.mesh(np.fliplr(vertices1), faces1)
    a2 = vv.subplot(122)
    vv.title('Classic')
    m2 = vv.mesh(np.fliplr(vertices2), faces2)
    a1.camera = a2.camera

    # visvis uses right-hand rule, gradient_direction param uses left-hand rule
    m1.cullFaces = m2.cullFaces = 'front'  # None, front or back

    vv.use().Run()


def donuts():
    """Return volume of two donuts based on a formula by Thomas Lewiner."""
    n = 48
    a = 2.5 / n * 8.0
    b = -1.25 * 8.0
    c = 16.0 - 1.85 * 1.85
    d = 64.0
    vol = np.empty((n, n, n), 'float32')
    for iz in range(n):
        z = iz * a + b
        z *= z
        zc = z + c
        for iy in range(n):
            y = iy * a + b
            y1 = y - 2
            y1 *= y1
            y2 = y + 2
            y2 *= y2
            for ix in range(n):
                x = ix * a + b
                x *= x
                x1 = x + y1 + zc
                x1 *= x1
                x2 = x + y2 + zc
                x2 *= x2
                vol[iz, iy, ix] = ((x1 - d * (x + y1)) * (x2 - d * (z + y2)))
    vol += 1025.0
    return vol


class Timer(object):
    """Context manager for timing execution speed of body."""

    def __init__(self, message):
        """Initialize timer."""
        print(message, end=' ')
        self.start = 0

    def __enter__(self):
        """Enter timer context."""
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit timer context."""
        print('took %1.0f ms' % (1000 * (time.time() - self.start)))


if __name__ == '__main__':
    main()
