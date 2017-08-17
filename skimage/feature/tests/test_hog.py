import os
import numpy as np
from scipy import ndimage as ndi
from skimage import color
from skimage import data, data_dir
from skimage import feature
from skimage import img_as_float
from skimage import draw
from numpy.testing import (assert_raises,
                           assert_almost_equal)


def test_hog_output_size():
    img = img_as_float(data.astronaut()[:256, :].mean(axis=2))

    fd = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(1, 1))

    assert len(fd) == 9 * (256 // 8) * (512 // 8)


def test_hog_output_correctness_l1_norm():
    img = color.rgb2gray(data.astronaut())
    correct_output = np.load(
        os.path.join(data_dir, 'astronaut_GRAY_hog_L1.npy'))

    output = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(3, 3), block_norm='L1',
                         feature_vector=True, transform_sqrt=False,
                         visualise=False)
    assert_almost_equal(output, correct_output)


def test_hog_output_correctness_l2hys_norm():
    img = color.rgb2gray(data.astronaut())
    correct_output = np.load(
        os.path.join(data_dir, 'astronaut_GRAY_hog_L2-Hys.npy'))

    output = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(3, 3), block_norm='L2-Hys',
                         feature_vector=True, transform_sqrt=False,
                         visualise=False)
    assert_almost_equal(output, correct_output)


def test_hog_image_size_cell_size_mismatch():
    image = data.camera()[:150, :200]
    fd = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(1, 1))
    assert len(fd) == 9 * (150 // 8) * (200 // 8)


def test_hog_color_image_unsupported_error():
    image = np.zeros((20, 20, 3))
    assert_raises(ValueError, feature.hog, image)


def test_hog_basic_orientations_and_data_types():
    # scenario:
    #  1) create image (with float values) where upper half is filled by
    #     zeros, bottom half by 100
    #  2) create unsigned integer version of this image
    #  3) calculate feature.hog() for both images, both with 'transform_sqrt'
    #     option enabled and disabled
    #  4) verify that all results are equal where expected
    #  5) verify that computed feature vector is as expected
    #  6) repeat the scenario for 90, 180 and 270 degrees rotated images

    # size of testing image
    width = height = 35

    image0 = np.zeros((height, width), dtype='float')
    image0[height // 2:] = 100

    for rot in range(4):
        # rotate by 0, 90, 180 and 270 degrees
        image_float = np.rot90(image0, rot)

        # create uint8 image from image_float
        image_uint8 = image_float.astype('uint8')

        (hog_float, hog_img_float) = feature.hog(
            image_float, orientations=4, pixels_per_cell=(8, 8),
            cells_per_block=(1, 1), visualise=True, transform_sqrt=False)
        (hog_uint8, hog_img_uint8) = feature.hog(
            image_uint8, orientations=4, pixels_per_cell=(8, 8),
            cells_per_block=(1, 1), visualise=True, transform_sqrt=False)
        (hog_float_norm, hog_img_float_norm) = feature.hog(
            image_float, orientations=4, pixels_per_cell=(8, 8),
            cells_per_block=(1, 1), visualise=True, transform_sqrt=True)
        (hog_uint8_norm, hog_img_uint8_norm) = feature.hog(
            image_uint8, orientations=4, pixels_per_cell=(8, 8),
            cells_per_block=(1, 1), visualise=True, transform_sqrt=True)

        # set to True to enable manual debugging with graphical output,
        # must be False for automatic testing
        if False:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(2, 3, 1)
            plt.imshow(image_float)
            plt.colorbar()
            plt.title('image')
            plt.subplot(2, 3, 2)
            plt.imshow(hog_img_float)
            plt.colorbar()
            plt.title('HOG result visualisation (float img)')
            plt.subplot(2, 3, 5)
            plt.imshow(hog_img_uint8)
            plt.colorbar()
            plt.title('HOG result visualisation (uint8 img)')
            plt.subplot(2, 3, 3)
            plt.imshow(hog_img_float_norm)
            plt.colorbar()
            plt.title('HOG result (transform_sqrt) visualisation (float img)')
            plt.subplot(2, 3, 6)
            plt.imshow(hog_img_uint8_norm)
            plt.colorbar()
            plt.title('HOG result (transform_sqrt) visualisation (uint8 img)')
            plt.show()

        # results (features and visualisation) for float and uint8 images must
        # be almost equal
        assert_almost_equal(hog_float, hog_uint8)
        assert_almost_equal(hog_img_float, hog_img_uint8)

        # resulting features should be almost equal when 'transform_sqrt' is enabled
        # or disabled (for current simple testing image)
        assert_almost_equal(hog_float, hog_float_norm, decimal=4)
        assert_almost_equal(hog_float, hog_uint8_norm, decimal=4)

        # reshape resulting feature vector to matrix with 4 columns (each
        # corresponding to one of 4 directions); only one direction should
        # contain nonzero values (this is manually determined for testing
        # image)
        actual = np.max(hog_float.reshape(-1, 4), axis=0)

        if rot in [0, 2]:
            # image is rotated by 0 and 180 degrees
            desired = [0, 0, 1, 0]
        elif rot in [1, 3]:
            # image is rotated by 90 and 270 degrees
            desired = [1, 0, 0, 0]
        else:
            raise Exception('Result is not determined for this rotation.')

        assert_almost_equal(actual, desired, decimal=2)


def test_hog_orientations_circle():
    # scenario:
    #  1) create image with blurred circle in the middle
    #  2) calculate feature.hog()
    #  3) verify that the resulting feature vector contains uniformly
    #     distributed values for all orientations, i.e. no orientation is
    #     lost or emphasized
    #  4) repeat the scenario for other 'orientations' option

    # size of testing image
    width = height = 100

    image = np.zeros((height, width))
    rr, cc = draw.circle(int(height / 2), int(width / 2), int(width / 3))
    image[rr, cc] = 100
    image = ndi.gaussian_filter(image, 2)

    for orientations in range(2, 15):
        (hog, hog_img) = feature.hog(image, orientations=orientations,
                                     pixels_per_cell=(8, 8),
                                     cells_per_block=(1, 1), visualise=True,
                                     transform_sqrt=False)

        # set to True to enable manual debugging with graphical output,
        # must be False for automatic testing
        if False:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.colorbar()
            plt.title('image_float')
            plt.subplot(1, 2, 2)
            plt.imshow(hog_img)
            plt.colorbar()
            plt.title('HOG result visualisation, '
                      'orientations=%d' % (orientations))
            plt.show()

        # reshape resulting feature vector to matrix with N columns (each
        # column corresponds to one direction),
        hog_matrix = hog.reshape(-1, orientations)

        # compute mean values in the resulting feature vector for each
        # direction, these values should be almost equal to the global mean
        # value (since the image contains a circle), i.e., all directions have
        # same contribution to the result
        actual = np.mean(hog_matrix, axis=0)
        desired = np.mean(hog_matrix)
        assert_almost_equal(actual, desired, decimal=1)


def test_hog_normalise_none_error_raised():
    img = np.array([1, 2, 3])
    assert_raises(ValueError, feature.hog, img, normalise=True)


def test_hog_block_normalization_incorrect_error():
    img = np.eye(4)
    assert_raises(ValueError, feature.hog, img, block_norm='Linf')


if __name__ == '__main__':
    np.testing.run_module_suite()
