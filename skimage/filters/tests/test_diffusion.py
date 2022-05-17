import numpy as np
from skimage.filters import diffusion_linear
from skimage.filters import diffusion_nonlinear_iso
from skimage.filters import diffusion_nonlinear_aniso
from skimage.metrics import structural_similarity as ssim
from numpy.testing import assert_equal
import pytest
from skimage import data, img_as_float
from skimage.filters import gaussian
from skimage.util.dtype import img_as_bool, img_as_uint


def Crop(img, border_x, border_y):
    return img[border_x: - border_x, border_y: - border_y]


camera_crop = Crop(data.camera(), 200, 200)  # image 117 x 117 pxls
cat_color = data.chelsea()[80: - 120, 120: - 190]  # image 100 x 141 x 3


@pytest.mark.parametrize('time_step', [0.25, 0.1])
@pytest.mark.parametrize('num_iters', [12, 25])
@pytest.mark.parametrize('scheme', ['aos', 'explicit'])
def test_gauss_lindiff_equal(time_step, num_iters, scheme):
    limit_diff = 0.99
    sigma = np.sqrt(2 * num_iters * time_step)
    img1 = gaussian(camera_crop, sigma)
    img2 = diffusion_linear(camera_crop, time_step=time_step,
                            num_iters=num_iters, scheme=scheme)
    assert ssim(img1, img2) > limit_diff


@pytest.mark.parametrize('scheme', ['aos', 'explicit'])
def test_min_max(scheme):
    limit_diff = 1/255
    print("OK")
    lin = diffusion_linear(camera_crop, scheme=scheme)
    iso = diffusion_nonlinear_iso(camera_crop, scheme=scheme)
    eed = diffusion_nonlinear_aniso(camera_crop, mode='eed', scheme=scheme)
    ced = diffusion_nonlinear_aniso(camera_crop, mode='ced', scheme=scheme)
    in_min = np.min(img_as_float(camera_crop))
    in_max = np.max(img_as_float(camera_crop))
    assert in_min <= np.min(lin) + limit_diff
    assert in_max >= np.max(lin) - limit_diff
    assert in_min <= np.min(iso) + limit_diff
    assert in_max >= np.max(iso) - limit_diff
    assert in_min <= np.min(eed) + limit_diff
    assert in_max >= np.max(eed) - limit_diff
    assert in_min <= np.min(ced) + limit_diff
    assert in_max >= np.max(ced) - limit_diff
    print("OK")

test_min_max('aos')
def getRelativeDifference(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return abs(a - b) / abs(max(a, b))


@pytest.mark.parametrize('diffusivity_type', ['perona-malik', 'charbonnier',
                         'exponential'])
@pytest.mark.parametrize('scheme', ['aos', 'explicit'])
def test_avg_intensity(diffusivity_type, scheme):
    # change in average grey value is expected to be less than 0.2%
    limit_diff = 0.002
    sum_cam = np.sum(cam)
    cam = camera_crop.astype(np.float64)
    assert getRelativeDifference(sum_cam, np.sum(
        diffusion_linear(cam, scheme=scheme))) < limit_diff
    assert getRelativeDifference(sum_cam, np.sum(
        diffusion_nonlinear_iso(cam, diffusivity_type=diffusivity_type,
                                scheme=scheme))) < limit_diff
    assert getRelativeDifference(sum_cam, np.sum(
        diffusion_nonlinear_aniso(cam, scheme=scheme))) < limit_diff
    assert getRelativeDifference(sum_cam, np.sum(
        diffusion_nonlinear_aniso(cam, mode='ced',
                                  scheme=scheme))) < limit_diff


@pytest.mark.parametrize('scheme', ['aos', 'explicit'])
@pytest.mark.parametrize('image', [img_as_bool(camera_crop),
                                   img_as_float(camera_crop),
                                   img_as_uint(camera_crop)])
def test_dtype(scheme, image):
    diffusion_linear(image, num_iters=2, scheme=scheme)
    diffusion_nonlinear_iso(image, num_iters=2, scheme=scheme)
    diffusion_nonlinear_aniso(image, mode="eed", num_iters=2, scheme=scheme)
    diffusion_nonlinear_aniso(image, mode="ced", num_iters=2, scheme=scheme)


@pytest.mark.parametrize('image', [camera_crop, cat_color])
@pytest.mark.parametrize('scheme', ['aos', 'explicit'])
def test_shape(image, scheme):
    in_shape = image.shape
    assert_equal(diffusion_linear(
        image, num_iters=2, scheme=scheme).shape, in_shape)
    assert_equal(diffusion_nonlinear_iso(
        image, num_iters=2, scheme=scheme).shape, in_shape)
    assert_equal(
        diffusion_nonlinear_aniso(image, mode='eed', num_iters=2,
                                  scheme=scheme).shape, in_shape)
    assert_equal(
        diffusion_nonlinear_aniso(image, mode='ced', num_iters=2,
                                  scheme=scheme).shape, in_shape)


def test_parameter_validity():
    invalid_shape = np.zeros((3, 3, 3, 3))
    #  linear
    with pytest.raises(ValueError):
        diffusion_linear(
            camera_crop, time_step=-1)
    with pytest.raises(ValueError):
        diffusion_linear(
            camera_crop, num_iters=-1)
    with pytest.raises(RuntimeError):
        diffusion_linear(
            invalid_shape)
    with pytest.raises(ValueError):
        diffusion_linear(
            camera_crop, scheme='xplicit')
    #  iso
    with pytest.raises(ValueError):
        diffusion_nonlinear_iso(
            camera_crop, lmbd=0)
    with pytest.raises(ValueError):
        diffusion_nonlinear_iso(
            camera_crop, time_step=-1)
    with pytest.raises(ValueError):
        diffusion_nonlinear_iso(
            camera_crop, num_iters=-1)
    with pytest.raises(RuntimeError):
        diffusion_nonlinear_iso(
            invalid_shape)
    with pytest.raises(ValueError):
        diffusion_nonlinear_iso(
            camera_crop, scheme='xplicit')
    #  aniso
    with pytest.raises(ValueError):
        diffusion_nonlinear_aniso(
            camera_crop, lmbd=0)
    with pytest.raises(ValueError):
        diffusion_nonlinear_aniso(
            camera_crop, time_step=-1)
    with pytest.raises(ValueError):
        diffusion_nonlinear_aniso(
            camera_crop, num_iters=-1)
    with pytest.raises(RuntimeError):
        diffusion_nonlinear_aniso(
            invalid_shape)
    with pytest.raises(ValueError):
        diffusion_nonlinear_aniso(
            camera_crop, scheme='xplicit')
    with pytest.raises(ValueError):
        diffusion_nonlinear_aniso(
            camera_crop, mode='cd')


@pytest.mark.parametrize('diff_type', ['perona-malik',
                         'charbonnier', 'exponential'])
def test_explicit_aos_equal(diff_type):
    limit_diff = 0.99
    time_step = 0.25  # maximal time_step stable for 'explicit'
    cam = camera_crop
    assert ssim(diffusion_linear(cam, time_step=time_step,
                                 scheme='explicit'),
                diffusion_linear(cam, time_step=time_step,
                                 scheme='aos')) > limit_diff
    assert ssim(diffusion_nonlinear_iso(cam,
                                        diffusivity_type=diff_type,
                                        time_step=time_step,
                                        scheme='explicit'),
                diffusion_nonlinear_iso(cam,
                                        diffusivity_type=diff_type,
                                        time_step=time_step,
                                        scheme='aos')) > limit_diff
    assert ssim(diffusion_nonlinear_aniso(cam, mode='eed',
                                          time_step=time_step,
                                          scheme='explicit'),
                diffusion_nonlinear_aniso(cam, mode='eed',
                                          time_step=time_step,
                                          scheme='aos')) > limit_diff
    assert ssim(diffusion_nonlinear_aniso(cam, mode='ced',
                                          time_step=time_step,
                                          scheme='explicit'),
                diffusion_nonlinear_aniso(cam, mode='ced',
                                          time_step=time_step,
                                          scheme='aos')) > limit_diff
