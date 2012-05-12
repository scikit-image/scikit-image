from numpy.testing import *
from unwrap2D import unwrap2D

import numpy as np
from numpy import outer, arange, ones, abs, empty, power, indices


def test_unwrap2D():
    nx, ny = 10, 10
    x = np.arange(nx)
    y = np.arange(ny)
    x.shape = (1,-1)
    y.shape = (-1,1)

    z = np.exp(1j*x*0.2*np.pi) * np.exp(1j*y*0.1*np.pi)
    phi_w = np.angle(z)
    mask = 0*np.ones((nx, ny), dtype = np.uint8)
    mask[4:6, 5:7] = 1
    phi = unwrap2D(phi_w.astype(np.float32), mask)
    return phi_w/(np.pi*2), np.asarray(phi)/(np.pi*2)
    

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
    p,p2 = test_unwrap2D()
    plt.clf()
    plt.imshow(p2, interpolation = 'nearest')
    plt.draw()
    plt.show()
