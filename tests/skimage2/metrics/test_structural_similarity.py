import math
import numpy as np
import pytest
from _skimage2.util._array_api import xp_assert_equal, assert_almost_equal

from _skimage2.metrics import structural_similarity
from _skimage2._shared.utils import _supported_float_type

from skimage import data

np.random.seed(5)
cam = data.camera()
sigma = 20.0
cam_noisy = np.clip(cam + sigma * np.random.randn(*cam.shape), 0, 255)
cam_noisy = cam_noisy.astype(cam.dtype)

np.random.seed(1234)


def test_structural_similarity_patch_range(xp):
    N = 51
    X = (np.random.rand(N, N) * 255).astype(np.uint8)
    Y = (np.random.rand(N, N) * 255).astype(np.uint8)

    X = xp.asarray(X)
    Y = xp.asarray(Y)

    assert structural_similarity(X, Y, data_range=255, win_size=N) < 0.1
    assert structural_similarity(X, X, data_range=255, win_size=N) == 1.0


def test_structural_similarity_image(xp):
    N = 100
    X = (np.random.rand(N, N) * 255).astype(np.uint8)
    Y = (np.random.rand(N, N) * 255).astype(np.uint8)

    X = xp.asarray(X)
    Y = xp.asarray(Y)

    S0 = structural_similarity(X, X, data_range=255, win_size=3)
    assert S0 == 1.0

    S1 = structural_similarity(X, Y, data_range=255, win_size=3)
    assert S1 < 0.3

    S2 = structural_similarity(X, Y, data_range=255, gaussian_weights=True)
    assert S2 < 0.3

    mssim0, S3 = structural_similarity(X, Y, data_range=255, full=True)
    assert S3.shape == X.shape
    mssim = structural_similarity(X, Y, data_range=255)
    assert mssim0 == mssim

    # structural_similarity of image with itself should be 1.0
    assert structural_similarity(X, X, data_range=255) == 1.0


# FIXME: Because we are forcing a random seed state, it is probably good to test
#        against a few seeds in case on seed gives a particularly bad example
@pytest.mark.parametrize('seed', [1, 2, 3, 5, 8, 13])
@pytest.mark.parametrize('dtype_str', ['float16', 'float32', 'float64'])
def test_structural_similarity_grad(seed, dtype_str, xp):
    N = 60
    # FIXME: This test is known to randomly fail on some systems (Mac OS X 10.6)
    #        And when testing tests in parallel. Therefore, we choose a few
    #        seeds that are known to work.
    #        The likely cause of this failure is that we are setting a hard
    #        threshold on the value of the gradient. Often the computed gradient
    #        is only slightly larger than what was measured.
    dtype_np = getattr(np, dtype_str)
    dtype_xp = getattr(xp, dtype_str)

    rng = np.random.default_rng(seed)
    X = rng.random((N, N)).astype(dtype_np, copy=False) * 255
    Y = rng.random((N, N)).astype(dtype_np, copy=False) * 255

    X = xp.asarray(X)
    Y = xp.asarray(Y)

    f = structural_similarity(X, Y, data_range=255)
    g = structural_similarity(X, Y, data_range=255, gradient=True)

    assert f < 0.05

    assert g[0] < 0.05
    assert xp.all(g[1] < 0.05)

    mssim, grad, s = structural_similarity(
        X, Y, data_range=255, gradient=True, full=True
    )

    assert s.dtype == _supported_float_type(dtype_xp, xp=xp)
    assert grad.dtype == _supported_float_type(dtype_xp, xp=xp)
    assert xp.all(grad < 0.05)


@pytest.mark.parametrize(
    'dtype_str', ['uint8', 'int32', 'float16', 'float32', 'float64']
)
def test_structural_similarity_dtype(dtype_str, xp):
    N = 30
    X = np.random.rand(N, N)
    Y = np.random.rand(N, N)

    X = xp.asarray(X)
    Y = xp.asarray(Y)
    dtype = getattr(xp, dtype_str)

    if xp.isdtype(dtype, ('integral', 'bool')):
#    if np.dtype(dtype).kind in 'iub':
        data_range = 255.0
        X = xp.astype(X * 255, dtype)
        Y = xp.astype(Y * 255, dtype)
    else:
        data_range = 1.0
        X = xp.astype(X, dtype, copy=False)
        Y = xp.astype(Y, dtype, copy=False)

    S1 = structural_similarity(X, Y, data_range=data_range)
    assert S1.dtype == xp.float64

    assert S1 < 0.1


@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1])
def test_structural_similarity_multichannel(channel_axis, xp):
    N = 100
    X = (np.random.rand(N, N) * 255).astype(np.uint8)
    Y = (np.random.rand(N, N) * 255).astype(np.uint8)

    X, Y = map(xp.asarray, (X, Y))
    data_range = xp.iinfo(xp.uint8).max

    S1 = structural_similarity(X, Y, data_range=data_range, win_size=3)

    # replicate across three channels.  should get identical value
    Xc = xp.tile(X[..., xp.newaxis], (1, 1, 3))
    Yc = xp.tile(Y[..., xp.newaxis], (1, 1, 3))

    # move channels from last position to specified channel_axis
    Xc, Yc = (xp.moveaxis(_arr, -1, channel_axis) for _arr in (Xc, Yc))

    S2 = structural_similarity(
        Xc, Yc, data_range=data_range, channel_axis=channel_axis, win_size=3
    )
    assert_almost_equal(S1, S2)

    # full case should return an image as well
    m, S3 = structural_similarity(
        Xc, Yc, data_range=data_range, channel_axis=channel_axis, full=True
    )
    assert S3.shape == Xc.shape

    # gradient case
    m, grad = structural_similarity(
        Xc, Yc, data_range=data_range, channel_axis=channel_axis, gradient=True
    )
    assert grad.shape == Xc.shape

    # full and gradient case
    m, grad, S3 = structural_similarity(
        Xc,
        Yc,
        data_range=data_range,
        channel_axis=channel_axis,
        full=True,
        gradient=True,
    )
    assert grad.shape == Xc.shape
    assert S3.shape == Xc.shape

    # fail if win_size exceeds any non-channel dimension
    with pytest.raises(ValueError):
        structural_similarity(
            Xc, Yc, data_range=data_range, win_size=7, channel_axis=None
        )


@pytest.mark.parametrize('dtype', [np.uint8, np.float32, np.float64])
@pytest.mark.parametrize('ndim', [1, 2, 3, 4])
def test_structural_similarity_nD(dtype, ndim, xp):
    # test 1D through 4D on small random arrays
    rng = np.random.default_rng(20260429)
    shape = (24 // ndim,) * ndim
    X = (rng.random(shape) * 255).astype(dtype)
    Y = (rng.random(shape) * 255).astype(dtype)

    X, Y = map(xp.asarray, (X, Y))

    mssim = structural_similarity(X, Y, win_size=3, data_range=255.0)
    assert mssim.dtype == xp.float64
    assert mssim < 0.05


def test_structural_similarity_multichannel_chelsea(xp):
    # color image example
    Xc = data.chelsea()
    sigma = 15.0
    Yc = xp.clip(xp.asarray(Xc + sigma * np.random.randn(*Xc.shape)), 0, 255)
    Xc = xp.asarray(Xc)
    Yc = xp.astype(Yc, Xc.dtype)

    # multichannel result should be mean of the individual channel results
    mssim = structural_similarity(Xc, Yc, data_range=255, channel_axis=-1)
    mssim_sep = [
        structural_similarity(Yc[..., c], Xc[..., c], data_range=255)
        for c in range(Xc.shape[-1])
    ]
    assert_almost_equal(mssim, xp.mean(xp.asarray(mssim_sep)))

    # structural_similarity of image with itself should be 1.0
    assert structural_similarity(Xc, Xc, data_range=255, channel_axis=-1) == 1.0


@pytest.mark.parametrize('gaussian_weights', [True, False])
def test_structural_similarity_dtype_insensitivity(gaussian_weights, xp):
    # Result of `structural_similarity` should be insensitive to the input dtype
    image_int = xp.arange(100, dtype=xp.int64)
    mssim_int = structural_similarity(
        image_int[1:],
        image_int[:-1],
        data_range=image_int.max(),
        gaussian_weights=gaussian_weights,
    )
    image_float = xp.astype(image_int, xp.float64)
    mssim_float = structural_similarity(
        image_float[1:],
        image_float[:-1],
        data_range=image_float.max(),
        gaussian_weights=gaussian_weights,
    )
    assert mssim_float == mssim_int
    assert mssim_float < 1.0


def test_gaussian_structural_similarity_vs_IPOL(xp):
    """Tests vs. imdiff result from the following IPOL article and code:
    https://www.ipol.im/pub/art/2011/g_lmii/.

    Notes
    -----
    To generate mssim_IPOL, we need a local copy of cam_noisy::

      from skimage import io
      io.imsave('/tmp/cam_noisy.png', cam_noisy)

    Then, we use the following command:
    $ ./imdiff -m mssim <path to camera.png>/camera.png /tmp/cam_noisy.png

    Values for current data.camera() calculated by Gregory Lee on Sep, 2020.
    Available at:
    https://github.com/scikit-image/scikit-image/pull/4913#issuecomment-700653165
    """
    mssim_IPOL = 0.357959091663361
    assert cam.dtype == np.uint8
    assert cam_noisy.dtype == np.uint8
    mssim = structural_similarity(
        xp.asarray(cam),
        xp.asarray(cam_noisy),
        data_range=xp.iinfo(xp.uint8).max,
        gaussian_weights=True,
        use_sample_covariance=False,
    )
    assert math.isclose(mssim, mssim_IPOL, abs_tol=1.5e-3)


@pytest.mark.parametrize(
    'dtype', [np.uint8, np.int32, np.float16, np.float32, np.float64]
)
def test_mssim_vs_legacy(dtype, xp):
    # check that ssim with default options matches skimage 0.17 result
    mssim_skimage_0pt17 = 0.3674518327910367
    assert cam.dtype == np.uint8
    assert cam_noisy.dtype == np.uint8
    mssim = structural_similarity(
        xp.asarray(cam.astype(dtype)),
        xp.asarray(cam_noisy.astype(dtype)),
        data_range=255
    )
    assert_almost_equal(xp.asarray(mssim), xp.asarray(mssim_skimage_0pt17))


def test_ssim_warns_about_data_range():
    mssim = structural_similarity(cam, cam_noisy, data_range=255)

    with pytest.warns(UserWarning, match='Inputs have mismatched dtypes'):
        _ = structural_similarity(
            cam,
            cam_noisy.astype(np.int32),
            data_range=np.iinfo(np.int32).max,
        )

    # Warn too when user supplies data_range
    with pytest.warns(UserWarning, match='Inputs have mismatched dtypes'):
        mssim_mixed = structural_similarity(
            cam, cam_noisy.astype(np.float32), data_range=255
        )

    assert_almost_equal(mssim, mssim_mixed)


@pytest.mark.parametrize('dtype_str', ['float16', 'float32', 'float64'])
def test_structural_similarity_small_image(dtype_str, xp):
    dtype = getattr(xp, dtype_str)
    X = xp.zeros((5, 5), dtype=dtype)
    # structural_similarity can be computed for small images if win_size is
    # a) odd and b) less than or equal to the images' smaller side
    assert structural_similarity(X, X, win_size=3, data_range=1.0) == 1.0
    assert structural_similarity(X, X, win_size=5, data_range=1.0) == 1.0
    # structural_similarity errors for small images if user doesn't specify
    # win_size
    with pytest.raises(ValueError):
        structural_similarity(X, X, data_range=1)


def test_structural_similarity_errors_without_data_range():
    X = np.zeros((64, 64))
    with pytest.raises(TypeError):
        structural_similarity(X, X)


def test_gaussian_weights_win_size_error():
    """win_size with gaussian_weights=True should raise ValueError (#7231)."""
    N = 100
    X = (np.random.rand(N, N) * 255).astype(np.uint8)
    Y = (np.random.rand(N, N) * 255).astype(np.uint8)

    with pytest.raises(
        ValueError, match="win_size cannot be specified when gaussian_weights"
    ):
        structural_similarity(X, Y, data_range=255, gaussian_weights=True, win_size=7)


def test_invalid_input():
    X = np.zeros((9, 9), dtype=np.float64)
    Y = np.zeros((8, 8), dtype=np.float64)
    with pytest.raises(ValueError, match="Input images must have the same dimensions"):
        structural_similarity(X, Y, data_range=1)
    with pytest.raises(ValueError, match="win_size exceeds image extent"):
        structural_similarity(X, X, data_range=1, win_size=X.shape[0] + 1)
    with pytest.raises(ValueError, match="K1 must be positive"):
        structural_similarity(X, X, data_range=1, K1=-0.1)
    with pytest.raises(ValueError, match="K2 must be positive"):
        structural_similarity(X, X, data_range=1, K2=-0.1)
    with pytest.raises(ValueError, match="sigma must be positive"):
        structural_similarity(X, X, data_range=1, sigma=-1.0)
