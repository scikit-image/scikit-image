import numpy as np
from skimage.filters._diffusion_nonlinear_aniso import diffusion_nonlinear_aniso
from skimage.filters._diffusion_linear import diffusion_linear
from skimage.filters._diffusion_nonlinear_iso import diffusion_nonlinear_iso

from numpy.testing import assert_equal
import pytest
from skimage import data

from skimage.filters import gaussian


def Crop(img, border_x, border_y):
    return img[border_x: - border_x, border_y: - border_y]


camera_crop = Crop(data.camera(), 200, 200)  # image 117 x 117 pxls
cat_color = data.chelsea()[80: - 120, 120: - 190]  # image 100 x 141 x 3


def getAverageRelativeDifference(a, b):
    return np.mean(np.abs(a-b)) / np.mean([np.mean(a), np.mean(b)])


"""
Implementation of diffusion filters - linear diffusion, 
nonlinear isotropic diffusion and nonlinear anisotropic diffusion.

Linear diffusion corresponds to a gaussian filter
and is present only for theoretical and consistency purposes. 

Nonlinear isotropic diffusion contains implementation of different diffusivity
types : Perona-Malik, Charbonniere, exponencial.

Nonlinear anisotropic diffusion has two modes ('eed', 'ced'), which correspond to 
Edge Enhancing Diffusion and Coherence Enhancing Diffusion.

All of the diffusion filters are computed by one of two schemes ('explicit', 'aos').
"""


@ pytest.mark.parametrize('time_step', [0.25, 0.1])
@ pytest.mark.parametrize('num_iters', [12, 25])
@ pytest.mark.parametrize('scheme', ['aos', 'explicit'])
def test_gauss_lindiff_equal(time_step, num_iters, scheme):
    limit_diff = 0.02
    sigma = np.sqrt(2*num_iters*time_step)
    img1 = 255*gaussian(camera_crop, sigma)
    img2 = diffusion_linear(camera_crop, time_step=time_step,
                            num_iters=num_iters, scheme=scheme)
    assert getAverageRelativeDifference(img1, img2) < limit_diff


@ pytest.mark.parametrize('scheme', ['aos', 'explicit'])
def test_min_max(scheme):
    limit_diff = 1
    lin = diffusion_linear(camera_crop, scheme=scheme)
    iso = diffusion_nonlinear_iso(camera_crop, scheme=scheme)
    eed = diffusion_nonlinear_aniso(camera_crop, mode='eed', scheme=scheme)
    ced = diffusion_nonlinear_aniso(camera_crop, mode='ced', scheme=scheme)
    in_min = np.min(camera_crop)
    in_max = np.max(camera_crop)
    assert in_min <= np.min(lin) + limit_diff
    assert in_max >= np.max(lin) - limit_diff
    assert in_min <= np.min(iso) + limit_diff
    assert in_max >= np.max(iso) - limit_diff
    assert in_min <= np.min(eed) + limit_diff
    assert in_max >= np.max(eed) - limit_diff
    assert in_min <= np.min(ced) + limit_diff
    assert in_max >= np.max(ced) - limit_diff


def getRelativeDifference(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return abs(a-b) / abs(max(a, b))


@ pytest.mark.parametrize('diffusivity_type', ['perona-malik', 'charbonnier', 'exponencial'])
@ pytest.mark.parametrize('scheme', ['aos', 'explicit'])
def test_sum(scheme, diffusivity_type):
    limit_diff = 0.002  # change in average grey value is expected to be less than 0.2%
    cam = camera_crop.astype(np.float64)
    assert getRelativeDifference(np.sum(cam), np.sum(
        diffusion_linear(cam, scheme=scheme))) < limit_diff
    assert getRelativeDifference(np.sum(cam), np.sum(
        diffusion_nonlinear_iso(cam, diffusivity_type=diffusivity_type, scheme=scheme))) < limit_diff
    assert getRelativeDifference(np.sum(cam), np.sum(
        diffusion_nonlinear_aniso(cam, scheme=scheme))) < limit_diff
    assert getRelativeDifference(np.sum(cam), np.sum(
        diffusion_nonlinear_aniso(cam, mode='ced', scheme=scheme))) < limit_diff


@ pytest.mark.parametrize('image', [camera_crop, cat_color])
@ pytest.mark.parametrize('scheme', ['aos', 'explicit'])
@ pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.uint8, np.uint16, np.uint32])
def test_dtype(image, dtype, scheme):
    img = image.astype(dtype)
    assert_equal(diffusion_linear(
        img, num_iters=2, scheme=scheme).dtype, dtype)
    assert_equal(diffusion_nonlinear_iso(
        img, num_iters=2, scheme=scheme).dtype, dtype)
    assert_equal(diffusion_nonlinear_aniso(
        img, mode="eed", num_iters=2, scheme=scheme).dtype, dtype)
    assert_equal(diffusion_nonlinear_aniso(
        img, mode="ced", num_iters=2, scheme=scheme).dtype, dtype)


@ pytest.mark.parametrize('image', [camera_crop, cat_color])
@ pytest.mark.parametrize('scheme', ['aos', 'explicit'])
def test_shape(image, scheme):
    in_shape = image.shape
    assert_equal(diffusion_linear(
        image, num_iters=2, scheme=scheme).shape, in_shape)
    assert_equal(diffusion_nonlinear_iso(
        image, num_iters=2, scheme=scheme).shape, in_shape)
    assert_equal(
        diffusion_nonlinear_aniso(image, mode='eed', num_iters=2, scheme=scheme).shape, in_shape)
    assert_equal(
        diffusion_nonlinear_aniso(image, mode='ced', num_iters=2, scheme=scheme).shape, in_shape)


@ pytest.mark.parametrize('scheme', ['as', 'ecplicit'])
@ pytest.mark.parametrize('mode', ['ed', 'cd'])
def test_parameter_validity(scheme, mode):
    with pytest.raises(ValueError):
        diffusion_linear(camera_crop, scheme=scheme)
    with pytest.raises(ValueError):
        diffusion_nonlinear_iso(camera_crop, scheme=scheme)
    with pytest.raises(ValueError):
        diffusion_nonlinear_aniso(camera_crop, scheme=scheme, mode=mode)
    with pytest.raises(ValueError):
        diffusion_nonlinear_aniso(
            camera_crop, mode='eed', time_step=-1, num_iters=-1, scheme='aos', alpha=0)
    with pytest.raises(ValueError):
        diffusion_nonlinear_aniso(
            camera_crop, mode='ced', time_step=-1, num_iters=-1, scheme='aos', alpha=0)
    with pytest.raises(ValueError):
        diffusion_nonlinear_iso(
            camera_crop, time_step=-1, num_iters=-1, scheme='aos', alpha=0)
    with pytest.raises(ValueError):
        diffusion_linear(
            camera_crop, time_step=-1, num_iters=-1, scheme='explicit', alpha=0)


@ pytest.mark.parametrize('diffusivity_type', ['perona-malik', 'charbonnier', 'exponencial'])
def test_explicit_aos_equal(diffusivity_type):
    limit_diff = 0.03  # 3% average relative difference
    cam = camera_crop.astype(np.float64)
    assert getAverageRelativeDifference(diffusion_linear(cam, scheme='explicit'),
                                        diffusion_linear(cam, scheme='aos')) < limit_diff
    assert getAverageRelativeDifference(diffusion_nonlinear_iso(cam, diffusivity_type=diffusivity_type, scheme='explicit'),
                                        diffusion_nonlinear_iso(cam, diffusivity_type=diffusivity_type, scheme='aos')) < limit_diff
    assert getAverageRelativeDifference(diffusion_nonlinear_aniso(cam, mode='eed', scheme='explicit'),
                                        diffusion_nonlinear_aniso(cam, mode='eed', scheme='aos')) < limit_diff
    assert getAverageRelativeDifference(diffusion_nonlinear_aniso(cam, mode='ced', scheme='explicit'),
                                        diffusion_nonlinear_aniso(cam, mode='ced', scheme='aos')) < limit_diff


# def Run(num_iter, file_name, scheme, mode):
#     test = io.imread(
#         "/home/alexandra/Documents/MUNI/Bakalarka/src/images/" + file_name)
#     # test = cv2.imread("/home/alexandra/Documents/MUNI/Bakalarka/src/images/" +
#     #                   file_name, cv2.IMREAD_GRAYSCALE).astype(np.float64)
#     # test = data.camera()
#     rho = 0
#     sigma = 2.5
#     alpha = 2.

#     if mode == 'ced':
#         rho = 6.
#         sig = 0.1
#         alpha = 0.01
#     print("aniso sum before : ", np.sum(test))

#     imgEED = diffusion_nonlinear_aniso(image=test, mode=mode, time_step=0.25,
#                                        num_iters=num_iter, scheme=scheme, sigma_ced=sigma, sigma_eed=sigma, rho=rho, alpha=alpha)
#     in_str = file_name.split(".")
#     print("aniso sum after : ", np.sum(imgEED))
#     # x, y = np.gradient(in_image2.astype(np.float64))
#     # print(x)

#     io.imsave("/home/alexandra/Documents/MUNI/Bakalarka/src/tests/sk_test/show/" +
#               in_str[0] + "_" + mode + str(num_iter) + "scheme_" + scheme + ".png", imgEED)


# def RunLinear(num_iter, file_name, scheme):
#     test = io.imread(
#         "/home/alexandra/Documents/MUNI/Bakalarka/src/images/" + file_name)
#     # test = cv2.imread("/home/alexandra/Documents/MUNI/Bakalarka/src/images/" + file_name, cv2.IMREAD_GRAYSCALE).astype(np.float64)
#     sigma = 5
#     alpha = 0.01
#     print("linear sum before : ", np.sum(test))

#     imgEED = diffusion_linear(image=test, time_step=0.25,
#                               num_iters=num_iter, scheme=scheme, sigma=sigma, alpha=alpha)
#     in_str = file_name.split(".")
#     print("linear sum after : ", np.sum(imgEED))

#     io.imsave("/home/alexandra/Documents/MUNI/Bakalarka/src/tests/sk_test/show/linear" +
#               in_str[0] + "_" + str(num_iter) + "scheme_" + scheme + ".png", imgEED)


# def RunIso(num_iter, file_name, scheme, dif_type):
#     test = io.imread(
#         "/home/alexandra/Documents/MUNI/Bakalarka/src/images/" + file_name)
#     # test = cv2.imread("/home/alexandra/Documents/MUNI/Bakalarka/src/images/" + file_name, cv2.IMREAD_GRAYSCALE).astype(np.float64)
#     sigma = 0.1
#     alpha = 2
#     t = 0.25
#     print("iso sum before : ", np.sum(test))

#     imgiso = diffusion_nonlinear_iso(
#         image=test, diffusivity_type=dif_type, time_step=t, num_iters=num_iter, scheme=scheme, sigma=sigma, alpha=alpha)
#     in_str = file_name.split(".")
#     print("iso sum after : ", np.sum(imgiso))
#     io.imsave("/home/alexandra/Documents/MUNI/Bakalarka/src/tests/sk_test/show/iso_" + in_str[0] + "_" + str(
#         num_iter) + "scheme_" + str(scheme) + "type_" + str(dif_type) + ".png", imgiso)


# Run(5, "orka.tif", 'aos', 'eed')
# Run(20, "shapes.tif", 'aos', 'eed')
# io.imsave("/home/alexandra/Documents/MUNI/Bakalarka/src/images/cat_color.png",
#           cat_color)

# RunLinear(20, 'cat_color.png', 'aos')
# RunIso(20, 'lena_noisy.png', 'aos', 'perona-malik')
