import numpy as np
from numpy.testing import run_module_suite, assert_array_equal, assert_raises

from skimage import img_as_ubyte, img_as_uint, img_as_float
from skimage import data, util
from skimage.morphology import cmorph, disk
from skimage.filter import rank


def test_random_sizes():
    # make sure the size is not a problem

    niter = 10
    elem = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
    for m, n in np.random.random_integers(1, 100, size=(10, 2)):
        mask = np.ones((m, n), dtype=np.uint8)

        image8 = np.ones((m, n), dtype=np.uint8)
        out8 = np.empty_like(image8)
        rank.mean(image=image8, selem=elem, mask=mask, out=out8,
                  shift_x=0, shift_y=0)
        assert_array_equal(image8.shape, out8.shape)
        rank.mean(image=image8, selem=elem, mask=mask, out=out8,
                  shift_x=+1, shift_y=+1)
        assert_array_equal(image8.shape, out8.shape)

        image16 = np.ones((m, n), dtype=np.uint16)
        out16 = np.empty_like(image8, dtype=np.uint16)
        rank.mean(image=image16, selem=elem, mask=mask, out=out16,
                  shift_x=0, shift_y=0)
        assert_array_equal(image16.shape, out16.shape)
        rank.mean(image=image16, selem=elem, mask=mask, out=out16,
                  shift_x=+1, shift_y=+1)
        assert_array_equal(image16.shape, out16.shape)

        rank.mean_percentile(image=image16, mask=mask, out=out16,
                             selem=elem, shift_x=0, shift_y=0, p0=.1, p1=.9)
        assert_array_equal(image16.shape, out16.shape)
        rank.mean_percentile(image=image16, mask=mask, out=out16,
                             selem=elem, shift_x=+1, shift_y=+1, p0=.1, p1=.9)
        assert_array_equal(image16.shape, out16.shape)


def test_compare_with_cmorph_dilate():
    # compare the result of maximum filter with dilate

    image = (np.random.random((100, 100)) * 256).astype(np.uint8)
    out = np.empty_like(image)
    mask = np.ones(image.shape, dtype=np.uint8)

    for r in range(1, 20, 1):
        elem = np.ones((r, r), dtype=np.uint8)
        rank.maximum(image=image, selem=elem, out=out, mask=mask)
        cm = cmorph._dilate(image=image, selem=elem)
        assert_array_equal(out, cm)


def test_compare_with_cmorph_erode():
    # compare the result of maximum filter with erode

    image = (np.random.random((100, 100)) * 256).astype(np.uint8)
    out = np.empty_like(image)
    mask = np.ones(image.shape, dtype=np.uint8)

    for r in range(1, 20, 1):
        elem = np.ones((r, r), dtype=np.uint8)
        rank.minimum(image=image, selem=elem, out=out, mask=mask)
        cm = cmorph._erode(image=image, selem=elem)
        assert_array_equal(out, cm)


def test_bitdepth():
    # test the different bit depth for rank16

    elem = np.ones((3, 3), dtype=np.uint8)
    out = np.empty((100, 100), dtype=np.uint16)
    mask = np.ones((100, 100), dtype=np.uint8)

    for i in range(5):
        image = np.ones((100, 100), dtype=np.uint16) * 255 * 2 ** i
        r = rank.mean_percentile(image=image, selem=elem, mask=mask,
                                 out=out, shift_x=0, shift_y=0, p0=.1, p1=.9)


def test_population():
    # check the number of valid pixels in the neighborhood

    image = np.zeros((5, 5), dtype=np.uint8)
    elem = np.ones((3, 3), dtype=np.uint8)
    out = np.empty_like(image)
    mask = np.ones(image.shape, dtype=np.uint8)

    rank.pop(image=image, selem=elem, out=out, mask=mask)
    r = np.array([[4, 6, 6, 6, 4],
                  [6, 9, 9, 9, 6],
                  [6, 9, 9, 9, 6],
                  [6, 9, 9, 9, 6],
                  [4, 6, 6, 6, 4]])
    assert_array_equal(r, out)


def test_structuring_element8():
    # check the output for a custom structuring element

    r = np.array([[0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 255, 0, 0, 0],
                  [0, 0, 255, 255, 255, 0],
                  [0, 0, 0, 255, 255, 0],
                  [0, 0, 0, 0, 0, 0]])

    # 8-bit
    image = np.zeros((6, 6), dtype=np.uint8)
    image[2, 2] = 255
    elem = np.asarray([[1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=np.uint8)
    out = np.empty_like(image)
    mask = np.ones(image.shape, dtype=np.uint8)

    rank.maximum(image=image, selem=elem, out=out, mask=mask,
                 shift_x=1, shift_y=1)
    assert_array_equal(r, out)

    # 16-bit
    image = np.zeros((6, 6), dtype=np.uint16)
    image[2, 2] = 255
    out = np.empty_like(image)

    rank.maximum(image=image, selem=elem, out=out, mask=mask,
                 shift_x=1, shift_y=1)
    assert_array_equal(r, out)


def test_pass_on_bitdepth():
    # should pass because data bitdepth is not too high for the function

    image = np.ones((100, 100), dtype=np.uint16) * 2 ** 11
    elem = np.ones((3, 3), dtype=np.uint8)
    out = np.empty_like(image)
    mask = np.ones(image.shape, dtype=np.uint8)


def test_inplace_output():
    # rank filters are not supposed to filter inplace

    selem = disk(20)
    image = (np.random.random((500, 500)) * 256).astype(np.uint8)
    out = image
    assert_raises(NotImplementedError, rank.mean, image, selem, out=out)


def test_compare_autolevels():
    # compare autolevel and percentile autolevel with p0=0.0 and p1=1.0
    # should returns the same arrays

    image = util.img_as_ubyte(data.camera())

    selem = disk(20)
    loc_autolevel = rank.autolevel(image, selem=selem)
    loc_perc_autolevel = rank.autolevel_percentile(image, selem=selem,
                                                   p0=.0, p1=1.)

    assert_array_equal(loc_autolevel, loc_perc_autolevel)


def test_compare_autolevels_16bit():
    # compare autolevel(16-bit) and percentile autolevel(16-bit) with p0=0.0
    # and p1=1.0 should returns the same arrays

    image = data.camera().astype(np.uint16) * 4

    selem = disk(20)
    loc_autolevel = rank.autolevel(image, selem=selem)
    loc_perc_autolevel = rank.autolevel_percentile(image, selem=selem,
                                                   p0=.0, p1=1.)

    assert_array_equal(loc_autolevel, loc_perc_autolevel)


def test_compare_ubyte_vs_float():

    # Create signed int8 image that and convert it to uint8
    image_uint = img_as_ubyte(data.camera()[:50, :50])
    image_float = img_as_float(image_uint)

    methods = ['autolevel', 'bottomhat', 'equalize', 'gradient', 'threshold',
               'subtract_mean', 'enhance_contrast', 'pop', 'tophat']

    for method in methods:
        func = getattr(rank, method)
        out_u = func(image_uint, disk(3))
        out_f = func(image_float, disk(3))
        assert_array_equal(out_u, out_f)


def test_compare_8bit_unsigned_vs_signed():
    # filters applied on 8-bit image ore 16-bit image (having only real 8-bit
    # of dynamic) should be identical

    # Create signed int8 image that and convert it to uint8
    image = img_as_ubyte(data.camera())
    image[image > 127] = 0
    image_s = image.astype(np.int8)
    image_u = img_as_ubyte(image_s)

    assert_array_equal(image_u, img_as_ubyte(image_s))

    methods = ['autolevel', 'bottomhat', 'equalize', 'gradient', 'maximum',
               'mean', 'subtract_mean', 'median', 'minimum', 'modal',
               'enhance_contrast', 'pop', 'threshold', 'tophat']

    for method in methods:
        func = getattr(rank, method)
        out_u = func(image_u, disk(3))
        out_s = func(image_s, disk(3))
        assert_array_equal(out_u, out_s)


def test_compare_8bit_vs_16bit():
    # filters applied on 8-bit image ore 16-bit image (having only real 8-bit
    # of dynamic) should be identical

    image8 = util.img_as_ubyte(data.camera())
    image16 = image8.astype(np.uint16)
    assert_array_equal(image8, image16)

    methods = ['autolevel', 'bottomhat', 'equalize', 'gradient', 'maximum',
               'mean', 'subtract_mean', 'median', 'minimum', 'modal',
               'enhance_contrast', 'pop', 'threshold', 'tophat']

    for method in methods:
        func = getattr(rank, method)
        f8 = func(image8, disk(3))
        f16 = func(image16, disk(3))
        assert_array_equal(f8, f16)


def test_trivial_selem8():
    # check that min, max and mean returns identity if structuring element
    # contains only central pixel

    image = np.zeros((5, 5), dtype=np.uint8)
    out = np.zeros_like(image)
    mask = np.ones_like(image, dtype=np.uint8)
    image[2, 2] = 255
    image[2, 3] = 128
    image[1, 2] = 16

    elem = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)
    rank.mean(image=image, selem=elem, out=out, mask=mask,
              shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    rank.minimum(image=image, selem=elem, out=out, mask=mask,
                 shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    rank.maximum(image=image, selem=elem, out=out, mask=mask,
                 shift_x=0, shift_y=0)
    assert_array_equal(image, out)


def test_trivial_selem16():
    # check that min, max and mean returns identity if structuring element
    # contains only central pixel

    image = np.zeros((5, 5), dtype=np.uint16)
    out = np.zeros_like(image)
    mask = np.ones_like(image, dtype=np.uint8)
    image[2, 2] = 255
    image[2, 3] = 128
    image[1, 2] = 16

    elem = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)
    rank.mean(image=image, selem=elem, out=out, mask=mask,
              shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    rank.minimum(image=image, selem=elem, out=out, mask=mask,
                 shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    rank.maximum(image=image, selem=elem, out=out, mask=mask,
                 shift_x=0, shift_y=0)
    assert_array_equal(image, out)


def test_smallest_selem8():
    # check that min, max and mean returns identity if structuring element
    # contains only central pixel

    image = np.zeros((5, 5), dtype=np.uint8)
    out = np.zeros_like(image)
    mask = np.ones_like(image, dtype=np.uint8)
    image[2, 2] = 255
    image[2, 3] = 128
    image[1, 2] = 16

    elem = np.array([[1]], dtype=np.uint8)
    rank.mean(image=image, selem=elem, out=out, mask=mask,
              shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    rank.minimum(image=image, selem=elem, out=out, mask=mask,
                 shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    rank.maximum(image=image, selem=elem, out=out, mask=mask,
                 shift_x=0, shift_y=0)
    assert_array_equal(image, out)


def test_smallest_selem16():
    # check that min, max and mean returns identity if structuring element
    # contains only central pixel

    image = np.zeros((5, 5), dtype=np.uint16)
    out = np.zeros_like(image)
    mask = np.ones_like(image, dtype=np.uint8)
    image[2, 2] = 255
    image[2, 3] = 128
    image[1, 2] = 16

    elem = np.array([[1]], dtype=np.uint8)
    rank.mean(image=image, selem=elem, out=out, mask=mask,
              shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    rank.minimum(image=image, selem=elem, out=out, mask=mask,
                 shift_x=0, shift_y=0)
    assert_array_equal(image, out)
    rank.maximum(image=image, selem=elem, out=out, mask=mask,
                 shift_x=0, shift_y=0)
    assert_array_equal(image, out)


def test_empty_selem():
    # check that min, max and mean returns zeros if structuring element is
    # empty

    image = np.zeros((5, 5), dtype=np.uint16)
    out = np.zeros_like(image)
    mask = np.ones_like(image, dtype=np.uint8)
    res = np.zeros_like(image)
    image[2, 2] = 255
    image[2, 3] = 128
    image[1, 2] = 16

    elem = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.uint8)

    rank.mean(image=image, selem=elem, out=out, mask=mask,
              shift_x=0, shift_y=0)
    assert_array_equal(res, out)
    rank.minimum(image=image, selem=elem, out=out, mask=mask,
                 shift_x=0, shift_y=0)
    assert_array_equal(res, out)
    rank.maximum(image=image, selem=elem, out=out, mask=mask,
                 shift_x=0, shift_y=0)
    assert_array_equal(res, out)


def test_otsu():
    # test the local Otsu segmentation on a synthetic image
    # (left to right ramp * sinus)

    test = np.tile([128, 145, 103, 127, 165, 83, 127, 185, 63, 127, 205, 43,
                    127, 225, 23, 127],
                   (16, 1))
    test = test.astype(np.uint8)
    res = np.tile([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], (16, 1))
    selem = np.ones((6, 6), dtype=np.uint8)
    th = 1 * (test >= rank.otsu(test, selem))
    assert_array_equal(th, res)


def test_entropy():
    #  verify that entropy is coherent with bitdepth of the input data

    selem = np.ones((16, 16), dtype=np.uint8)
    # 1 bit per pixel
    data = np.tile(np.asarray([0, 1]), (100, 100)).astype(np.uint8)
    assert(np.max(rank.entropy(data, selem)) == 1)

    # 2 bit per pixel
    data = np.tile(np.asarray([[0, 1], [2, 3]]), (10, 10)).astype(np.uint8)
    assert(np.max(rank.entropy(data, selem)) == 2)

    # 3 bit per pixel
    data = np.tile(
        np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]]), (10, 10)).astype(np.uint8)
    assert(np.max(rank.entropy(data, selem)) == 3)

    # 4 bit per pixel
    data = np.tile(
        np.reshape(np.arange(16), (4, 4)), (10, 10)).astype(np.uint8)
    assert(np.max(rank.entropy(data, selem)) == 4)

    # 6 bit per pixel
    data = np.tile(
        np.reshape(np.arange(64), (8, 8)), (10, 10)).astype(np.uint8)
    assert(np.max(rank.entropy(data, selem)) == 6)

    # 8-bit per pixel
    data = np.tile(
        np.reshape(np.arange(256), (16, 16)), (10, 10)).astype(np.uint8)
    assert(np.max(rank.entropy(data, selem)) == 8)

    # 12 bit per pixel
    selem = np.ones((64, 64), dtype=np.uint8)
    data = np.tile(
        np.reshape(np.arange(4096), (64, 64)), (2, 2)).astype(np.uint16)
    assert(np.max(rank.entropy(data, selem)) == 12)

    # make sure output is of dtype double
    out = rank.entropy(data, np.ones((16, 16), dtype=np.uint8))
    assert out.dtype == np.double


def test_selem_dtypes():

    image = np.zeros((5, 5), dtype=np.uint8)
    out = np.zeros_like(image)
    mask = np.ones_like(image, dtype=np.uint8)
    image[2, 2] = 255
    image[2, 3] = 128
    image[1, 2] = 16

    for dtype in (np.uint8, np.uint16, np.int32, np.int64,
                  np.float32, np.float64):
        elem = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=dtype)
        rank.mean(image=image, selem=elem, out=out, mask=mask,
                  shift_x=0, shift_y=0)
        assert_array_equal(image, out)
        rank.mean_percentile(image=image, selem=elem, out=out, mask=mask,
                             shift_x=0, shift_y=0)
        assert_array_equal(image, out)


def test_16bit():
    image = np.zeros((21, 21), dtype=np.uint16)
    selem = np.ones((3, 3), dtype=np.uint8)

    for bitdepth in range(17):
        value = 2 ** bitdepth - 1
        image[10, 10] = value
        assert rank.minimum(image, selem)[10, 10] == 0
        assert rank.maximum(image, selem)[10, 10] == value
        assert rank.mean(image, selem)[10, 10] == int(value / selem.size)


def test_bilateral():
    image = np.zeros((21, 21), dtype=np.uint16)
    selem = np.ones((3, 3), dtype=np.uint8)

    image[10, 10] = 1000
    image[10, 11] = 1010
    image[10, 9] = 900

    assert rank.mean_bilateral(image, selem, s0=1, s1=1)[10, 10] == 1000
    assert rank.pop_bilateral(image, selem, s0=1, s1=1)[10, 10] == 1
    assert rank.mean_bilateral(image, selem, s0=11, s1=11)[10, 10] == 1005
    assert rank.pop_bilateral(image, selem, s0=11, s1=11)[10, 10] == 2


def test_percentile_min():
    # check that percentile p0 = 0 is identical to local min
    img = data.camera()
    img16 = img.astype(np.uint16)
    selem = disk(15)
    # check for 8bit
    img_p0 = rank.percentile(img, selem=selem, p0=0)
    img_min = rank.minimum(img, selem=selem)
    assert_array_equal(img_p0, img_min)
    # check for 16bit
    img_p0 = rank.percentile(img16, selem=selem, p0=0)
    img_min = rank.minimum(img16, selem=selem)
    assert_array_equal(img_p0, img_min)


def test_percentile_max():
    # check that percentile p0 = 1 is identical to local max
    img = data.camera()
    img16 = img.astype(np.uint16)
    selem = disk(15)
    # check for 8bit
    img_p0 = rank.percentile(img, selem=selem, p0=1.)
    img_max = rank.maximum(img, selem=selem)
    assert_array_equal(img_p0, img_max)
    # check for 16bit
    img_p0 = rank.percentile(img16, selem=selem, p0=1.)
    img_max = rank.maximum(img16, selem=selem)
    assert_array_equal(img_p0, img_max)


def test_percentile_median():
    # check that percentile p0 = 0.5 is identical to local median
    img = data.camera()
    img16 = img.astype(np.uint16)
    selem = disk(15)
    # check for 8bit
    img_p0 = rank.percentile(img, selem=selem, p0=.5)
    img_max = rank.median(img, selem=selem)
    assert_array_equal(img_p0, img_max)
    # check for 16bit
    img_p0 = rank.percentile(img16, selem=selem, p0=.5)
    img_max = rank.median(img16, selem=selem)
    assert_array_equal(img_p0, img_max)

def test_sum():
    # check the number of valid pixels in the neighborhood

    image8 = np.array([[0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0]], dtype=np.uint8)
    image16 = 400*np.array([[0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0]], dtype=np.uint16)
    elem = np.ones((3, 3), dtype=np.uint8)
    out8 = np.empty_like(image8)
    out16 = np.empty_like(image16)
    mask = np.ones(image8.shape, dtype=np.uint8)

    r =  np.array([[1, 2, 3, 2, 1],
           [2, 4, 6, 4, 2],
           [3, 6, 9, 6, 3],
           [2, 4, 6, 4, 2],
           [1, 2, 3, 2, 1]], dtype=np.uint8)
    rank.sum(image=image8, selem=elem, out=out8, mask=mask)
    assert_array_equal(r, out8)
    rank.sum_percentile(image=image8, selem=elem, out=out8, mask=mask,p0=.0,p1=1.)
    assert_array_equal(r, out8)
    rank.sum_bilateral(image=image8, selem=elem, out=out8, mask=mask,s0=255,s1=255)
    assert_array_equal(r, out8)

    r = 400* np.array([[1, 2, 3, 2, 1],
           [2, 4, 6, 4, 2],
           [3, 6, 9, 6, 3],
           [2, 4, 6, 4, 2],
           [1, 2, 3, 2, 1]], dtype=np.uint16)
    rank.sum(image=image16, selem=elem, out=out16, mask=mask)
    assert_array_equal(r, out16)
    rank.sum_percentile(image=image16, selem=elem, out=out16, mask=mask,p0=.0,p1=1.)
    assert_array_equal(r, out16)
    rank.sum_bilateral(image=image16, selem=elem, out=out16, mask=mask,s0=1000,s1=1000)
    assert_array_equal(r, out16)


if __name__ == "__main__":
    run_module_suite()
