from numpy.testing import *
from unwrap import unwrap

import numpy as np

def test_unwrap2D():
    nx, ny = 16, 32
    x = np.arange(nx)
    y = np.arange(ny)
    x.shape = (-1,1)
    y.shape = (1,-1)

    z = np.exp(1j*x*0.2*np.pi) * np.exp(1j*y*0.1*np.pi)
    phi_w = np.angle(z)
    phi = unwrap(phi_w, 
                 wrap_around_axis_0 = False)

    mask = 0*np.ones((nx, ny), dtype = np.uint8)
    mask[4:16, 4:16] = 1
    phi_w_ma = np.ma.array(phi_w, dtype = np.float32, mask = mask)
    phi_ma = unwrap(phi_w_ma)

    return (phi/(np.pi*2),
            phi_w/(np.pi*2), phi/(np.pi*2),
            phi_w_ma/(np.pi*2), phi_ma/(np.pi*2),)
    
def test_unwrap3D():
    x, y, z = np.ogrid[:8, :12, :4]

    phi = 2*np.pi*(x*0.2 + y*0.1 + z*0.05)
    phi_wrapped = np.angle(np.exp(1j*phi))
    phi_unwrapped = unwrap(phi_wrapped)

    mask = np.zeros_like(phi, dtype = np.uint8)
    mask[4:6, 4:6, 1:3] = 1
    phi_wrapped_masked = np.ma.array(phi_wrapped, dtype = np.float32, mask = mask)
    phi_unwrapped_masked = unwrap(phi_wrapped_masked)

    s = np.round(phi_unwrapped[0,0,0]/(2*np.pi))
    assert_array_almost_equal(phi/(2*np.pi), phi_unwrapped/(2*np.pi) - s, decimal = 5)
    
    assert_array_almost_equal(phi + 2*np.pi*s, phi_unwrapped_masked, decimal = 5)

    return (phi/(np.pi*2),
            phi_wrapped/(np.pi*2), phi_unwrapped/(np.pi*2),
            phi_wrapped_masked/(np.pi*2), phi_unwrapped_masked/(np.pi*2),)



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
    
    p0,p1,p2,p3,p4 = test_unwrap2D()
    plt.figure(1)
    plt.clf()
    plt.subplot(322)
    plt.imshow(p0,interpolation = 'nearest')
    plt.subplot(323)
    plt.imshow(p1,interpolation = 'nearest')
    plt.subplot(324)
    plt.imshow(p2, interpolation = 'nearest')
    plt.subplot(325)
    plt.imshow(p3, interpolation = 'nearest')
    plt.subplot(326)
    plt.imshow(p4, interpolation = 'nearest')
    plt.draw()


    p0,p1,p2,p3,p4 = test_unwrap3D()
    plt.figure(2)
    plt.clf()
    plt.subplot(322)
    plt.imshow(p0[:,:,0],interpolation = 'nearest')
    plt.subplot(323)
    plt.imshow(p1[:,:,0],interpolation = 'nearest')
    plt.subplot(324)
    plt.imshow(p2[:,:,0], interpolation = 'nearest')
    plt.subplot(325)
    plt.imshow(p3[:,:,0], interpolation = 'nearest')
    plt.subplot(326)
    plt.imshow(p4[:,:,0], interpolation = 'nearest')
    plt.draw()
    plt.show()
