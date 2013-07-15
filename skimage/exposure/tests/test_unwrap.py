from __future__ import print_function, division

import numpy as np
from numpy.testing import (run_module_suite, assert_array_almost_equal,
                           assert_almost_equal, assert_array_equal)
import warnings

from skimage.exposure import unwrap


def assert_phase_almost_equal(a, b, *args, **kwargs):
    '''An assert_almost_equal insensitive to phase shifts of n*2*pi.'''
    shift = 2 * np.pi * np.round((b.mean() - a.mean()) / (2 * np.pi))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print('assert_phase_allclose, abs', np.max(np.abs(a - (b - shift))))
        print('assert_phase_allclose, rel',
              np.max(np.abs((a - (b - shift)) / a)))
    if np.ma.isMaskedArray(a):
        assert np.ma.isMaskedArray(b)
        assert_array_equal(a.mask, b.mask)
        au = np.asarray(a)
        bu = np.asarray(b)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print('assert_phase_allclose, no mask, abs',
                  np.max(np.abs(au - (bu - shift))))
            print('assert_phase_allclose, no mask, rel',
                  np.max(np.abs((au - (bu - shift)) / au)))
    assert_array_almost_equal(a + shift, b, *args, **kwargs)


def check_unwrap(image, mask=None):
    image_wrapped = np.angle(np.exp(1j * image))
    if not mask is None:
        print('Testing a masked image')
        image = np.ma.array(image, mask=mask)
        image_wrapped = np.ma.array(image_wrapped, mask=mask)
    image_unwrapped = unwrap(image_wrapped)
    assert_phase_almost_equal(image_unwrapped, image)


def test_unwrap_2d():
    x, y = np.ogrid[:8, :16]
    image = 2 * np.pi * (x * 0.2 + y * 0.1)
    yield check_unwrap, image
    mask = np.zeros(image.shape, dtype=np.bool)
    mask[4:6, 4:8] = True
    yield check_unwrap, image, mask


def test_unwrap_3d():
    x, y, z = np.ogrid[:8, :12, :4]
    image = 2 * np.pi * (x * 0.2 + y * 0.1 + z * 0.05)
    yield check_unwrap, image
    mask = np.zeros(image.shape, dtype=np.bool)
    mask[4:6, 4:6, 1:3] = True
    yield check_unwrap, image, mask


def check_wrap_around(ndim, axis):
    # create a ramp, but with the last pixel along axis equalling the first
    elements = 100
    ramp = np.linspace(0, 12 * np.pi, elements)
    ramp[-1] = ramp[0]
    image = ramp.reshape(tuple([elements if n == axis else 1
                                for n in range(ndim)]))
    image_wrapped = np.angle(np.exp(1j * image))

    index_first = tuple([0] * ndim)
    index_last = tuple([-1 if n == axis else 0 for n in range(ndim)])
    # unwrap the image without wrap around
    image_unwrap_no_wrap_around = unwrap(image_wrapped)
    print('endpoints without wrap_around:',
          image_unwrap_no_wrap_around[index_first],
          image_unwrap_no_wrap_around[index_last])
    # without wrap around, the endpoints of the image should differ
    assert abs(image_unwrap_no_wrap_around[index_first]
               - image_unwrap_no_wrap_around[index_last]) > np.pi
    # unwrap the image with wrap around
    wrap_around = [n == axis for n in range(ndim)]
    image_unwrap_wrap_around = unwrap(image_wrapped, wrap_around)
    print('endpoints with wrap_around:',
          image_unwrap_wrap_around[index_first],
          image_unwrap_wrap_around[index_last])
    # with wrap around, the endpoints of the image should be equal
    assert_almost_equal(image_unwrap_wrap_around[index_first],
                        image_unwrap_wrap_around[index_last])


def test_wrap_around():
    for ndim in (2, 3):
        for axis in range(ndim):
            yield check_wrap_around, ndim, axis


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
