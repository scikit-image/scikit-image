import numpy as np
from numpy.testing import run_module_suite, assert_array_equal, assert_raises

from skimage import data
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

        rank.percentile_mean(image=image16, mask=mask, out=out16,
            selem=elem, shift_x=0, shift_y=0, p0=.1, p1=.9)
        assert_array_equal(image16.shape, out16.shape)
        rank.percentile_mean(image=image16, mask=mask, out=out16,
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
        cm = cmorph.dilate(image=image, selem=elem)
        assert_array_equal(out, cm)


def test_compare_with_cmorph_erode():
    # compare the result of maximum filter with erode

    image = (np.random.random((100, 100)) * 256).astype(np.uint8)
    out = np.empty_like(image)
    mask = np.ones(image.shape, dtype=np.uint8)

    for r in range(1, 20, 1):
        elem = np.ones((r, r), dtype=np.uint8)
        rank.minimum(image=image, selem=elem, out=out, mask=mask)
        cm = cmorph.erode(image=image, selem=elem)
        assert_array_equal(out, cm)


def test_bitdepth():
    # test the different bit depth for rank16

    elem = np.ones((3, 3), dtype=np.uint8)
    out = np.empty((100, 100), dtype=np.uint16)
    mask = np.ones((100, 100), dtype=np.uint8)

    for i in range(5):
        image = np.ones((100, 100),dtype=np.uint16) * 255 * 2**i
        r = rank.percentile_mean(image=image, selem=elem, mask=mask,
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

    r = np.array([[  0,   0,   0,   0,   0,   0],
                  [  0,   0,   0,   0,   0,   0],
                  [  0,   0, 255,   0,   0,   0],
                  [  0,   0, 255, 255, 255,   0],
                  [  0,   0,   0, 255, 255,   0],
                  [  0,   0,   0,   0,   0,   0]])

    # 8bit
    image = np.zeros((6, 6), dtype=np.uint8)
    image[2, 2] = 255
    elem = np.asarray([[1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=np.uint8)
    out = np.empty_like(image)
    mask = np.ones(image.shape, dtype=np.uint8)

    rank.maximum(image=image, selem=elem, out=out, mask=mask,
        shift_x=1, shift_y=1)
    assert_array_equal(r, out)

    # 16bit
    image = np.zeros((6, 6), dtype=np.uint16)
    image[2, 2] = 255
    out = np.empty_like(image)

    rank.maximum(image=image, selem=elem, out=out, mask=mask,
        shift_x=1, shift_y=1)
    assert_array_equal(r, out)


def test_fail_on_bitdepth():
    # should fail because data bitdepth is too high for the function

    image = np.ones((100, 100), dtype=np.uint16) * 2**12
    elem = np.ones((3, 3), dtype=np.uint8)
    out = np.empty_like(image)
    mask = np.ones(image.shape, dtype=np.uint8)
    assert_raises(ValueError, rank.percentile_mean, image=image,
        selem=elem, out=out, mask=mask, shift_x=0, shift_y=0)

def test_pass_on_bitdepth():
    # should pass because data bitdepth is not too high for the function

    image = np.ones((100, 100), dtype=np.uint16) * 2**11
    elem = np.ones((3, 3), dtype=np.uint8)
    out = np.empty_like(image)
    mask = np.ones(image.shape, dtype=np.uint8)


def test_inplace_output():
    # rank filters are not supposed to filter inplace

    selem = disk(20)
    image = (np.random.random((500,500))*256).astype(np.uint8)
    out = image
    assert_raises(NotImplementedError, rank.mean, image, selem, out=out)


def test_compare_autolevels():
    # compare autolevel and percentile autolevel with p0=0.0 and p1=1.0
    # should returns the same arrays

    image = data.camera()

    selem = disk(20)
    loc_autolevel = rank.autolevel(image, selem=selem)
    loc_perc_autolevel = rank.percentile_autolevel(image, selem=selem,
        p0=.0, p1=1.)

    assert_array_equal(loc_autolevel, loc_perc_autolevel)


def test_compare_autolevels_16bit():
    # compare autolevel(16bit) and percentile autolevel(16bit) with p0=0.0 and
    # p1=1.0 should returns the same arrays

    image = data.camera().astype(np.uint16) * 4

    selem = disk(20)
    loc_autolevel = rank.autolevel(image, selem=selem)
    loc_perc_autolevel = rank.percentile_autolevel(image, selem=selem,
        p0=.0, p1=1.)

    assert_array_equal(loc_autolevel, loc_perc_autolevel)


def test_compare_8bit_vs_16bit():
    # filters applied on 8bit image ore 16bit image (having only real 8bit of
    # dynamic) should be identical

    image8 = data.camera()
    image16 = image8.astype(np.uint16)
    assert_array_equal(image8, image16)

    methods = ['autolevel', 'bottomhat', 'equalize', 'gradient', 'maximum',
               'mean', 'meansubstraction', 'median', 'minimum', 'modal',
               'morph_contr_enh', 'pop', 'threshold',  'tophat']

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
    image[2,2] = 255
    image[2,3] = 128
    image[1,2] = 16

    elem = np.array([[0, 0, 0], [0, 1, 0],[0, 0, 0]], dtype=np.uint8)
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
    image[2,2] = 255
    image[2,3] = 128
    image[1,2] = 16

    elem = np.array([[0, 0, 0], [0, 1, 0],[0, 0, 0]], dtype=np.uint8)
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
    image[2,2] = 255
    image[2,3] = 128
    image[1,2] = 16

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
    image[2,2] = 255
    image[2,3] = 128
    image[1,2] = 16

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
    # check that min, max and mean returns zeros if structuring element is empty

    image = np.zeros((5, 5), dtype=np.uint16)
    out = np.zeros_like(image)
    mask = np.ones_like(image, dtype=np.uint8)
    res = np.zeros_like(image)
    image[2,2] = 255
    image[2,3] = 128
    image[1,2] = 16

    elem = np.array([[0,0,0],[0,0,0]], dtype=np.uint8)

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
    #
    pass

def test_entropy():
    #  verify that entropy is coherent with bitdepth of the input data

    selem = np.ones((16,16), dtype=np.uint8)
    # 1 bit per pixel
    data = np.tile(np.asarray([0,1]),(100,100)).astype(np.uint8)
    assert(np.max(rank.entropy(data,selem))==10)

    # 2 bit per pixel
    data = np.tile(np.asarray([[0,1],[2,3]]),(10,10)).astype(np.uint8)
    assert(np.max(rank.entropy(data,selem))==20)

    # 3 bit per pixel
    data = np.tile(np.asarray([[0,1,2,3],[4,5,6,7]]),(10,10)).astype(np.uint8)
    assert(np.max(rank.entropy(data,selem))==30)

    # 4 bit per pixel
    data = np.tile(np.reshape(np.arange(16),(4,4)),(10,10)).astype(np.uint8)
    assert(np.max(rank.entropy(data,selem))==40)

    # 6 bit per pixel
    data = np.tile(np.reshape(np.arange(64),(8,8)),(10,10)).astype(np.uint8)
    assert(np.max(rank.entropy(data,selem))==60)

    # 8 bit per pixel
    data = np.tile(np.reshape(np.arange(256),(16,16)),(10,10)).astype(np.uint8)
    assert(np.max(rank.entropy(data,selem))==80)

    # 12 bit per pixel
    selem = np.ones((64,64), dtype=np.uint8)
    data = np.tile(np.reshape(np.arange(4096),(64,64)),(2,2)).astype(np.uint16)
    assert(np.max(rank.entropy(data,selem))==12000)


if __name__ == "__main__":
    run_module_suite()
