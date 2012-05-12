from numpy.testing import *
from unwrap2D import unwrap2D

import numpy as np
from numpy import outer, arange, ones, abs, empty, power, indices

import numpy.ma as ma

def test_unwrap2D():
    nx, ny = 32, 32
    x = np.arange(nx)
    y = np.arange(ny)
    x.shape = (1,-1)
    y.shape = (-1,1)

    z = np.exp(1j*x*0.2*np.pi) * np.exp(1j*y*0.1*np.pi)
    phi_w = np.angle(z)
    phi = unwrap2D(phi_w)

    mask = 0*np.ones((nx, ny), dtype = np.uint8)
    mask[4:16, 4:16] = 1
    phi_w_ma = ma.array(phi_w, dtype = np.float32, mask = mask)
    phi_ma = unwrap2D(phi_w_ma)

    return (phi_w/(np.pi*2), phi/(np.pi*2),
            phi_w_ma/(np.pi*2), phi_ma/(np.pi*2),)
    

# class test_unwrap(TestCase):
#     def test_simple2d(self, level=1):
#         grid = outer(ones(64), arange(-32,32)) + \
#                1.j * outer(arange(-32,32), ones(64))
#         pgrid = abs(grid)
#         wr_grid = normalize_angle(pgrid)
#         uw_grid = unwrap2D(wr_grid)
#         uw_grid += (pgrid[32,32] - uw_grid[32,32])
        
#         assert_array_almost_equal(pgrid, uw_grid, decimal=5)
    
#     def test_simple3d(self):
#         grid = indices((64,64,64))
#         grid[0] -= 32
#         grid[1] -= 32
#         grid[2] -= 32
#         # get distance of each point in the grid from 0
#         grid = power(power(grid, 2.0).sum(axis=0), 0.5)
#         wr_grid = normalize_angle(grid)
#         uw_grid = unwrap3D(wr_grid)
#         uw_grid += (grid[32,32,32] - uw_grid[32,32,32])
#         assert_array_almost_equal(grid, uw_grid, decimal=5)

if __name__=="__main__":
    #NumpyTest().run()
    import matplotlib.pyplot as plt
    p1,p2,p3,p4 = test_unwrap2D()
    plt.clf()
    plt.subplot(221)
    plt.imshow(p1,interpolation = 'nearest')
    plt.subplot(222)
    plt.imshow(p2, interpolation = 'nearest')
    plt.subplot(223)
    plt.imshow(p3, interpolation = 'nearest')
    plt.subplot(224)
    plt.imshow(p4, interpolation = 'nearest')
    plt.draw()
    plt.show()
