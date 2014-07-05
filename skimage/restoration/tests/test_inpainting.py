import numpy as np
from numpy.testing import assert_equal, assert_raises
from skimage.restoration import inpaint_fmm


def test_basic():
    expected = np.array(
        [[186, 187, 188, 185, 183, 185, 185, 176, 160, 129, 93, 51, 18,  8, 10],
         [186, 187, 187, 184, 182, 184, 180, 159, 127,  77, 32, 18, 16, 13, 13],
         [185, 185, 185, 184, 184, 183, 174, 146, 107,  59, 18, 10, 13, 12, 13],
         [186, 185, 184, 186, 185, 183, 174, 153, 121,  79, 41, 21, 13, 12, 14],
         [186, 185, 185, 185, 185, 184, 172, 150, 117,  71, 22, 18, 13, 13, 14],
         [187, 187, 187, 187, 186, 184, 173, 146, 107,  34, 20, 18, 14, 14, 15],
         [187, 187, 187, 188, 188, 186, 156, 132,  62,  32, 19, 15, 14, 13, 13],
         [189, 188, 188, 189, 187, 183, 153, 108,  61,  28, 15, 12, 15, 13, 11],
         [190, 189, 190, 190, 182, 172, 122,  81,  59,  24, 13, 11, 12, 10, 10],
         [191, 191, 192, 189, 174, 151,  97,  58,  33,  19, 13, 11, 10,  9, 10],
         [187, 188, 191, 184, 171, 128,  77,  41,  24,  15, 12,  9,  9,  9, 10],
         [185, 187, 190, 184, 168, 117,  58,  27,  18,  13, 12,  9, 10, 10, 10],
         [188, 191, 191, 189, 170,  98,  29,  12,  13,  10, 10,  9, 10,  9,  9],
         [192, 196, 194, 174, 140,  76,  19,  10,  16,  13, 11, 10, 11,  9,  9],
         [189, 196, 193, 159, 113,  58,  13,   6,  13,  12, 11, 10, 10,  8,  9]],
        dtype=np.uint8)

    mask = np.zeros(expected.shape, dtype=np.uint8)
    mask[3:12, 3:12] = 1

    image = expected.copy()
    image[mask == 1] = 0

    assert_equal(inpaint_fmm(image, mask, radius=5), expected)


def test_invalid_input():
    # Invalid shapes of image and mask
    assert_raises(ValueError, inpaint_fmm, np.zeros((8, 8), dtype=np.uint8),
                  np.zeros((8, 9), dtype=np.bool))

    # Invalid image
    assert_raises(TypeError, inpaint_fmm, np.zeros((8, 8), dtype=np.double),
                  np.zeros((8, 8), dtype=np.bool))

    # Invalid radius
    assert_raises(ValueError, inpaint_fmm, np.zeros((8, 8), dtype=np.uint8),
                  np.zeros((8, 8), dtype=np.bool), 0)
    assert_raises(ValueError, inpaint_fmm, np.zeros((8, 8), dtype=np.uint8),
                  np.zeros((8, 8), dtype=np.bool), -1)


if __name__ == "__main__":
    np.testing.run_module_suite()
