import numpy as np
import pytest
from scipy import ndimage as ndi
from numpy.testing import assert_allclose, assert_array_equal

import skimage as ski
from skimage.morphology import footprint_rectangle, mirror_footprint, pad_footprint
from skimage._shared.testing import fetch

import skimage2.morphology._grayscale_operators as gray


@pytest.fixture
def cam_image():
    return np.ascontiguousarray(ski.data.camera()[64:112, 64:96])


@pytest.fixture
def cell3d_image():
    return np.ascontiguousarray(ski.data.cells3d()[30:48, 0, 20:36, 20:32])


gray_operators = (
    gray.erosion,
    gray.dilation,
    gray.opening,
    gray.closing,
    gray.white_tophat,
    gray.black_tophat,
)


class TestMorphology:
    @pytest.mark.parametrize(
        "footprint_args",
        [
            ("square", lambda n: footprint_rectangle((n, n))),
            ("diamond", ski.morphology.diamond),
            ("disk", ski.morphology.disk),
            ("star", ski.morphology.star),
        ],
    )
    @pytest.mark.parametrize("size", list(range(1, 4)))
    @pytest.mark.parametrize(
        "func",
        [
            gray.erosion,
            gray.opening,
            gray.white_tophat,
        ],
    )
    def test_reproduce_skimage_data_not_mirrored(self, footprint_args, size, func):
        # Test that `erosion`, `opening`, and `white_tophat` can
        # reproduce data in `gray_morph_output.npz`
        image = ski.color.rgb2gray(ski.data.coffee())
        image = ski.transform.downscale_local_mean(image, (20, 20))
        image = ski.util.img_as_ubyte(image)

        footprint_name, footprint_func = footprint_args
        key = f'{footprint_name}_{size}_{func.__name__}'
        data = dict(np.load(fetch('data/gray_morph_output.npz')))
        expected = data[key]
        footprint = footprint_func(size)

        result = func(image, footprint, mode="reflect")
        np.testing.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "footprint_args",
        [
            ("square", lambda n: footprint_rectangle((n, n))),
            ("diamond", ski.morphology.diamond),
            ("disk", ski.morphology.disk),
            ("star", ski.morphology.star),
        ],
    )
    @pytest.mark.parametrize("size", list(range(1, 4)))
    @pytest.mark.parametrize(
        "func",
        [
            gray.dilation,
            gray.closing,
            gray.black_tophat,
        ],
    )
    def test_reproduce_skimage_data_mirrored(self, footprint_args, size, func):
        # Test that `dilation`, `closing`, and `black_tophat` can
        # reproduce data in `gray_morph_output.npz`
        image = ski.color.rgb2gray(ski.data.coffee())
        image = ski.transform.downscale_local_mean(image, (20, 20))
        image = ski.util.img_as_ubyte(image)

        footprint_name, footprint_func = footprint_args
        key = f'{footprint_name}_{size}_{func.__name__}'
        data = dict(np.load(fetch('data/gray_morph_output.npz')))
        expected = data[key]
        footprint = footprint_func(size)

        # Difference to the test above (`test_reproduce_skimage_data_not_mirrored`)
        footprint = pad_footprint(footprint, pad_end=False)
        footprint = mirror_footprint(footprint)

        result = func(image, footprint, mode="reflect")
        np.testing.assert_equal(result, expected)

    def test_gray_closing_extensive(self):
        img = ski.data.coins()
        footprint = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])

        # Default mode="ignore" is extensive
        result = gray.closing(img, footprint=footprint)
        assert np.all(result >= img)
        result = gray.closing(img, footprint=footprint, mode="ignore")
        assert np.all(result >= img)

        # mode="reflect" (v1.x default) is not extensive
        result_default = gray.closing(img, footprint=footprint, mode="reflect")
        assert not np.all(result_default >= img)

    def test_gray_opening_anti_extensive(self):
        img = ski.data.coins()
        footprint = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])

        # Default mode="ignore" is anti-extensive
        result_ignore = gray.opening(img, footprint=footprint)
        assert np.all(result_ignore <= img)
        result_ignore = gray.opening(img, footprint=footprint, mode="ignore")
        assert np.all(result_ignore <= img)

        # mode="reflect" (v1.x default) is not anti-extensive
        result_default = gray.opening(img, footprint=footprint, mode="reflect")
        assert not np.all(result_default <= img)

    @pytest.mark.parametrize("func", gray_operators)
    @pytest.mark.parametrize("mode", gray._SUPPORTED_MODES)
    def test_supported_mode(self, func, mode):
        img = np.ones((10, 10))
        func(img, mode=mode)

    @pytest.mark.parametrize("func", gray_operators)
    @pytest.mark.parametrize("mode", ["", "symmetric", 3, None])
    def test_unsupported_mode(self, func, mode):
        img = np.ones((10, 10))
        with pytest.raises(ValueError, match="unsupported mode"):
            func(img, mode=mode)


class TestEccentricStructuringElements:
    def setup_class(self):
        self.black_pixel = 255 * np.ones((6, 6), dtype=np.uint8)
        self.black_pixel[2, 2] = 0
        self.white_pixel = 255 - self.black_pixel
        self.footprints = [
            footprint_rectangle((2, 2)),
            footprint_rectangle((2, 1)),
            footprint_rectangle((1, 2)),
        ]

    def test_dilate_erode_symmetry(self):
        for footprint in self.footprints:
            eroded = gray.erosion(self.black_pixel, footprint=footprint)

            # Dilation mirrors footprint internally so that closing is extensive
            # and opening anti-extensive. To receive a symmetric result, we need
            # to use an asymmetric footprint. Also pad to odd-size before
            # mirroring so that correct side is padded with 0.
            asym_footprint = mirror_footprint(pad_footprint(footprint, pad_end=False))
            dilated = gray.dilation(self.white_pixel, footprint=asym_footprint)

            assert np.all(eroded == (255 - dilated))

    def test_open_black_pixel(self):
        for s in self.footprints:
            gray_open = gray.opening(self.black_pixel, s)
            assert np.all(gray_open == self.black_pixel)

    def test_close_white_pixel(self):
        for s in self.footprints:
            gray_close = gray.closing(self.white_pixel, s)
            assert np.all(gray_close == self.white_pixel)

    def test_open_white_pixel(self):
        for s in self.footprints:
            assert np.all(gray.opening(self.white_pixel, s) == 0)

    def test_close_black_pixel(self):
        for s in self.footprints:
            assert np.all(gray.closing(self.black_pixel, s) == 255)

    def test_white_tophat_white_pixel(self):
        for s in self.footprints:
            tophat = gray.white_tophat(self.white_pixel, s)
            assert np.all(tophat == self.white_pixel)

    def test_black_tophat_black_pixel(self):
        for s in self.footprints:
            tophat = gray.black_tophat(self.black_pixel, s)
            assert np.all(tophat == self.white_pixel)

    def test_white_tophat_black_pixel(self):
        for s in self.footprints:
            tophat = gray.white_tophat(self.black_pixel, s)
            assert np.all(tophat == 0)

    def test_black_tophat_white_pixel(self):
        for s in self.footprints:
            tophat = gray.black_tophat(self.white_pixel, s)
            assert np.all(tophat == 0)


@pytest.mark.parametrize("func", gray_operators)
def test_default_footprint(func):
    strel = ski.morphology.diamond(radius=1)
    image = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        np.uint8,
    )
    im_expected = func(image, strel)
    im_test = func(image)
    assert_array_equal(im_expected, im_test)


def test_3d_fallback_default_footprint():
    # 3x3x3 cube inside a 7x7x7 image:
    image = np.zeros((7, 7, 7), bool)
    image[2:-2, 2:-2, 2:-2] = 1

    opened = gray.opening(image)

    # expect a "hyper-cross" centered in the 5x5x5:
    image_expected = np.zeros((7, 7, 7), dtype=bool)
    image_expected[2:5, 2:5, 2:5] = ndi.generate_binary_structure(3, 1)
    assert_array_equal(opened, image_expected)


@pytest.mark.parametrize("func", [gray.closing, gray.opening])
def test_3d_fallback_cube_footprint(func):
    # 3x3x3 cube inside a 7x7x7 image:
    image = np.zeros((7, 7, 7), bool)
    image[2:-2, 2:-2, 2:-2] = 1

    cube = np.ones((3, 3, 3), dtype=np.uint8)

    new_image = func(image, cube)
    assert_array_equal(new_image, image)


def test_3d_fallback_white_tophat():
    image = np.zeros((7, 7, 7), dtype=bool)
    image[2, 2:4, 2:4] = 1
    image[3, 2:5, 2:5] = 1
    image[4, 3:5, 3:5] = 1

    new_image = gray.white_tophat(image)
    footprint = ndi.generate_binary_structure(3, 1)
    image_expected = ndi.white_tophat(image.view(dtype=np.uint8), footprint=footprint)
    assert_array_equal(new_image, image_expected)


def test_3d_fallback_black_tophat():
    image = np.ones((7, 7, 7), dtype=bool)
    image[2, 2:4, 2:4] = 0
    image[3, 2:5, 2:5] = 0
    image[4, 3:5, 3:5] = 0

    new_image = gray.black_tophat(image)
    footprint = ndi.generate_binary_structure(3, 1)
    image_expected = ndi.black_tophat(image.view(dtype=np.uint8), footprint=footprint)
    assert_array_equal(new_image, image_expected)


def test_2d_ndimage_equivalence():
    image = np.zeros((9, 9), np.uint8)
    image[2:-2, 2:-2] = 128
    image[3:-3, 3:-3] = 196
    image[4, 4] = 255

    opened = gray.opening(image)
    closed = gray.closing(image)

    footprint = ndi.generate_binary_structure(2, 1)
    ndimage_opened = ndi.grey_opening(image, footprint=footprint)
    ndimage_closed = ndi.grey_closing(image, footprint=footprint)

    assert_array_equal(opened, ndimage_opened)
    assert_array_equal(closed, ndimage_closed)


# float test images
im = np.array(
    [
        [0.55, 0.72, 0.6, 0.54, 0.42],
        [0.65, 0.44, 0.89, 0.96, 0.38],
        [0.79, 0.53, 0.57, 0.93, 0.07],
        [0.09, 0.02, 0.83, 0.78, 0.87],
        [0.98, 0.8, 0.46, 0.78, 0.12],
    ]
)

eroded = np.array(
    [
        [0.55, 0.44, 0.54, 0.42, 0.38],
        [0.44, 0.44, 0.44, 0.38, 0.07],
        [0.09, 0.02, 0.53, 0.07, 0.07],
        [0.02, 0.02, 0.02, 0.78, 0.07],
        [0.09, 0.02, 0.46, 0.12, 0.12],
    ]
)

dilated = np.array(
    [
        [0.72, 0.72, 0.89, 0.96, 0.54],
        [0.79, 0.89, 0.96, 0.96, 0.96],
        [0.79, 0.79, 0.93, 0.96, 0.93],
        [0.98, 0.83, 0.83, 0.93, 0.87],
        [0.98, 0.98, 0.83, 0.78, 0.87],
    ]
)

opened = np.array(
    [
        [0.55, 0.55, 0.54, 0.54, 0.42],
        [0.55, 0.44, 0.54, 0.44, 0.38],
        [0.44, 0.53, 0.53, 0.78, 0.07],
        [0.09, 0.02, 0.78, 0.78, 0.78],
        [0.09, 0.46, 0.46, 0.78, 0.12],
    ]
)

closed = np.array(
    [
        [0.72, 0.72, 0.72, 0.54, 0.54],
        [0.72, 0.72, 0.89, 0.96, 0.54],
        [0.79, 0.79, 0.79, 0.93, 0.87],
        [0.79, 0.79, 0.83, 0.78, 0.87],
        [0.98, 0.83, 0.78, 0.78, 0.78],
    ]
)


def test_float():
    assert_allclose(gray.erosion(im), eroded)
    assert_allclose(gray.dilation(im), dilated)
    assert_allclose(gray.opening(im), opened)
    assert_allclose(gray.closing(im), closed)


def test_uint16():
    im16, eroded16, dilated16, opened16, closed16 = map(
        ski.util.img_as_uint, [im, eroded, dilated, opened, closed]
    )
    assert_allclose(gray.erosion(im16), eroded16)
    assert_allclose(gray.dilation(im16), dilated16)
    assert_allclose(gray.opening(im16), opened16)
    assert_allclose(gray.closing(im16), closed16)


def test_discontiguous_out_array():
    image = np.array([[5, 6, 2], [7, 2, 2], [3, 5, 1]], np.uint8)
    out_array_big = np.zeros((5, 5), np.uint8)
    out_array = out_array_big[::2, ::2]
    expected_dilation = np.array(
        [
            [7, 0, 6, 0, 6],
            [0, 0, 0, 0, 0],
            [7, 0, 7, 0, 2],
            [0, 0, 0, 0, 0],
            [7, 0, 5, 0, 5],
        ],
        np.uint8,
    )
    expected_erosion = np.array(
        [
            [5, 0, 2, 0, 2],
            [0, 0, 0, 0, 0],
            [2, 0, 2, 0, 1],
            [0, 0, 0, 0, 0],
            [3, 0, 1, 0, 1],
        ],
        np.uint8,
    )
    gray.dilation(image, out=out_array)
    assert_array_equal(out_array_big, expected_dilation)
    gray.erosion(image, out=out_array)
    assert_array_equal(out_array_big, expected_erosion)


def test_1d_erosion():
    image = np.array([1, 2, 3, 2, 1])
    expected = np.array([1, 1, 2, 1, 1])
    eroded = gray.erosion(image)
    assert_array_equal(eroded, expected)


@pytest.mark.parametrize("func", gray_operators)
@pytest.mark.parametrize("nrows", [3, 7, 11])
@pytest.mark.parametrize("ncols", [3, 7, 11])
@pytest.mark.parametrize("decomposition", ['separable', 'sequence'])
def test_rectangle_decomposition(cam_image, func, nrows, ncols, decomposition):
    """Validate footprint decomposition for various shapes.

    comparison is made to the case without decomposition.
    """
    footprint_ndarray = footprint_rectangle((nrows, ncols), decomposition=None)
    footprint = footprint_rectangle((nrows, ncols), decomposition=decomposition)
    expected = func(cam_image, footprint=footprint_ndarray)
    out = func(cam_image, footprint=footprint)
    assert_array_equal(expected, out)


@pytest.mark.parametrize("func", gray_operators)
@pytest.mark.parametrize("radius", (2, 3))
@pytest.mark.parametrize("decomposition", ['sequence'])
def test_diamond_decomposition(cam_image, func, radius, decomposition):
    """Validate footprint decomposition for various shapes.

    comparison is made to the case without decomposition.
    """
    footprint_ndarray = ski.morphology.diamond(radius, decomposition=None)
    footprint = ski.morphology.diamond(radius, decomposition=decomposition)
    expected = func(cam_image, footprint=footprint_ndarray)
    out = func(cam_image, footprint=footprint)
    assert_array_equal(expected, out)


@pytest.mark.parametrize("func", gray_operators)
@pytest.mark.parametrize("m", (0, 1, 3, 5))
@pytest.mark.parametrize("n", (0, 1, 2, 3))
@pytest.mark.parametrize("decomposition", ['sequence'])
@pytest.mark.filterwarnings(
    "ignore:.*falling back to decomposition='separable':UserWarning"
)
def test_octagon_decomposition(cam_image, func, m, n, decomposition):
    """Validate footprint decomposition for various shapes.

    comparison is made to the case without decomposition.
    """
    if m == 0 and n == 0:
        with pytest.raises(ValueError):
            ski.morphology.octagon(m, n, decomposition=decomposition)
    else:
        footprint_ndarray = ski.morphology.octagon(m, n, decomposition=None)
        footprint = ski.morphology.octagon(m, n, decomposition=decomposition)
        expected = func(cam_image, footprint=footprint_ndarray)
        out = func(cam_image, footprint=footprint)
        assert_array_equal(expected, out)


@pytest.mark.parametrize("func", gray_operators)
@pytest.mark.parametrize("shape", [(5, 5, 5), (5, 5, 7)])
@pytest.mark.parametrize("decomposition", ['separable', 'sequence'])
def test_cube_decomposition(cell3d_image, func, shape, decomposition):
    """Validate footprint decomposition for various shapes.

    comparison is made to the case without decomposition.
    """
    footprint_ndarray = footprint_rectangle(shape, decomposition=None)
    footprint = footprint_rectangle(shape, decomposition=decomposition)
    expected = func(cell3d_image, footprint=footprint_ndarray)
    out = func(cell3d_image, footprint=footprint)
    assert_array_equal(expected, out)


@pytest.mark.parametrize("func", gray_operators)
@pytest.mark.parametrize("radius", (3,))
@pytest.mark.parametrize("decomposition", ['sequence'])
def test_octahedron_decomposition(cell3d_image, func, radius, decomposition):
    """Validate footprint decomposition for various shapes.

    comparison is made to the case without decomposition.
    """
    footprint_ndarray = ski.morphology.octahedron(radius, decomposition=None)
    footprint = ski.morphology.octahedron(radius, decomposition=decomposition)
    expected = func(cell3d_image, footprint=footprint_ndarray)
    out = func(cell3d_image, footprint=footprint)
    assert_array_equal(expected, out)
