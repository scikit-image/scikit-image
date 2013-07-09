import numpy as np
from numpy.testing import run_module_suite, TestCase, assert_array_almost_equal

from skimage.exposure import unwrap

def test_unwrap2D():
    x, y = np.ogrid[:8, :16]
    phi = 2*np.pi*(x*0.2 + y*0.1)
    phi_wrapped = np.angle(np.exp(1j*phi))
    phi_unwrapped = unwrap(phi_wrapped)

    s = np.round(phi_unwrapped[0,0]/(2*np.pi))
    assert_array_almost_equal(phi, phi_unwrapped - s*2*np.pi)

def test_unwrap2D_masked():
    x, y = np.ogrid[:8, :16]
    phi = 2*np.pi*(x*0.2 + y*0.1)

    mask = np.zeros_like(phi, dtype = np.uint8)
    mask[4:6, 4:8] = 1

    phi_wrapped = np.angle(np.exp(1j*phi))
    phi_wrapped_masked = np.ma.array(phi_wrapped, dtype = np.float32, mask = mask)
    phi_unwrapped_masked = unwrap(phi_wrapped_masked)

    s = np.round(phi_unwrapped_masked[0,0]/(2*np.pi))
    assert_array_almost_equal(phi + 2*np.pi*s, phi_unwrapped_masked)

def test_unwrap3D():
    x, y, z = np.ogrid[:8, :12, :4]
    phi = 2*np.pi*(x*0.2 + y*0.1 + z*0.05)
    phi_wrapped = np.angle(np.exp(1j*phi))
    phi_unwrapped = unwrap(phi_wrapped)

    s = np.round(phi_unwrapped[0,0]/(2*np.pi))
    assert_array_almost_equal(phi, phi_unwrapped - s*2*np.pi)

def test_unwrap3D_masked():
    x, y, z = np.ogrid[:8, :12, :4]
    phi = 2*np.pi*(x*0.2 + y*0.1 + z*0.05)
    phi_wrapped = np.angle(np.exp(1j*phi))
    mask = np.zeros_like(phi, dtype = np.uint8)
    mask[4:6, 4:6, 1:3] = 1
    phi_wrapped_masked = np.ma.array(phi_wrapped, dtype = np.float32, mask = mask)
    phi_unwrapped_masked = unwrap(phi_wrapped_masked)

    s = np.round(phi_unwrapped_masked[0,0,0]/(2*np.pi))
    assert_array_almost_equal(phi + 2*np.pi*s, phi_unwrapped_masked)

def unwrap_plots():

    x, y = np.ogrid[:32, :32]
    phi = 2*np.pi*(x*0.2 + y*0.1)

    #phi = 1*np.arctan2(x-14.3, y-6.3) - 2*np.arctan2(x-18.3, y-22.1)

    phi[8,8] = np.NaN

    phi_wrapped = np.angle(np.exp(1j*phi))
    phi_unwrapped = unwrap(phi_wrapped,
                           #wrap_around_axis_0 = True,
                           #wrap_around_axis_1 = True,
                           )

    mask = np.zeros_like(phi, dtype = np.uint8)
    #mask[10:22, 4:10] = 1
    phi_wrapped_masked = np.ma.array(phi_wrapped, dtype = np.float32, mask = mask)
    phi_unwrapped_masked = unwrap(phi_wrapped_masked)

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    plt.gray()
    plt.subplot(221)
    plt.imshow(phi, interpolation = 'nearest')
    plt.subplot(222)
    plt.imshow(phi_wrapped, interpolation = 'nearest')
    plt.subplot(223)
    plt.imshow(phi_unwrapped, interpolation = 'nearest')
    plt.subplot(224)
    plt.imshow(phi_unwrapped_masked, interpolation = 'nearest')

    plt.draw()
    plt.show()

if __name__=="__main__":
    run_module_suite()
