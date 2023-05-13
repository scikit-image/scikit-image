import os

import numpy as np
import imageio
from skimage import data_dir
from skimage.io.collection import ImageCollection, MultiImage, alphanumeric_key
from skimage.io import reset_plugins, imread

from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_allclose

import pytest


try:
    has_pooch = True
except ModuleNotFoundError:
    has_pooch = False


@pytest.fixture(scope="session")
def random_gif_path(tmpdir_factory):
    """Create "random.gif" once per session and return its path."""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 255, (24, 25, 14, 3), dtype=np.uint8)
    tmp_path = str(tmpdir_factory.mktemp("session-data").join("random.gif"))
    imageio.v3.imwrite(tmp_path, img)
    return tmp_path


def test_string_split():
    test_string = 'z23a'
    test_str_result = ['z', 23, 'a']
    assert_equal(alphanumeric_key(test_string), test_str_result)


def test_string_sort():
    filenames = ['f9.10.png', 'f9.9.png', 'f10.10.png', 'f10.9.png',
                 'e9.png', 'e10.png', 'em.png']
    expected_filenames = ['e9.png', 'e10.png', 'em.png', 'f9.9.png',
                          'f9.10.png', 'f10.9.png', 'f10.10.png']
    sorted_filenames = sorted(filenames, key=alphanumeric_key)
    assert_equal(expected_filenames, sorted_filenames)

def test_imagecollection_input():
    """Test function for ImageCollection. The new behavior (implemented
    in 0.16) allows the `pattern` argument to accept a list of strings
    as the input.

    Notes
    -----
        If correct, `images` will receive three images.
    """
    # Ensure that these images are part of the legacy datasets
    # this means they will always be available in the user's install
    # regardless of the availability of pooch
    pattern = [os.path.join(data_dir, pic)
               for pic in ['coffee.png',
                           'chessboard_GRAY.png',
                           'rocket.jpg']]
    images = ImageCollection(pattern)
    assert len(images) == 3


class TestImageCollection():
    pattern = [os.path.join(data_dir, pic)
               for pic in ['brick.png', 'color.png']]

    pattern_matched = [os.path.join(data_dir, pic)
                       for pic in ['brick.png', 'moon.png']]

    def setup_method(self):
        reset_plugins()
        # Generic image collection with images of different shapes.
        self.images = ImageCollection(self.pattern)
        # Image collection with images having shapes that match.
        self.images_matched = ImageCollection(self.pattern_matched)
        # Same images as a collection of frames
        self.frames_matched = MultiImage(self.pattern_matched)

    def test_len(self):
        assert len(self.images) == 2

    def test_getitem(self):
        num = len(self.images)
        for i in range(-num, num):
            assert isinstance(self.images[i], np.ndarray)
        assert_allclose(self.images[0],
                        self.images[-num])

        def return_img(n):
            return self.images[n]
        with testing.raises(IndexError):
            return_img(num)
        with testing.raises(IndexError):
            return_img(-num - 1)

    def test_slicing(self):
        assert type(self.images[:]) is ImageCollection
        assert len(self.images[:]) == 2
        assert len(self.images[:1]) == 1
        assert len(self.images[1:]) == 1
        assert_allclose(self.images[0], self.images[:1][0])
        assert_allclose(self.images[1], self.images[1:][0])
        assert_allclose(self.images[1], self.images[::-1][0])
        assert_allclose(self.images[0], self.images[::-1][1])

    def test_files_property(self):
        assert isinstance(self.images.files, list)

        def set_files(f):
            self.images.files = f
        with testing.raises(AttributeError):
            set_files('newfiles')

    @pytest.mark.skipif(not has_pooch, reason="needs pooch to download data")
    def test_custom_load_func_sequence(self, random_gif_path):
        def reader(frameno):
            return imread(random_gif_path, mode="RGB")[frameno, ...]

        ic = ImageCollection(range(24), load_func=reader)
        # the length of ic should be that of the given load_pattern sequence
        assert len(ic) == 24
        # GIF file has frames of size 25x14 with 3 channels (RGB)
        assert ic[0].shape == (25, 14, 3)

    @pytest.mark.skipif(not has_pooch, reason="needs pooch to download data")
    def test_custom_load_func_w_kwarg(self, random_gif_path):

        def load_fn(f, step):
            img = imread(f)
            seq = [frame for frame in img]
            return seq[::step]

        ic = ImageCollection(random_gif_path, load_func=load_fn, step=3)
        # Each file should map to one image (array).
        assert len(ic) == 1
        # GIF file has 24 frames, so 24 / 3 equals 8.
        assert len(ic[0]) == 8

    def test_custom_load_func(self):

        def load_fn(x):
            return x

        ic = ImageCollection(os.pathsep.join(self.pattern), load_func=load_fn)
        assert_equal(ic[0], self.pattern[0])

    def test_concatenate(self):
        array = self.images_matched.concatenate()
        expected_shape = (len(self.images_matched),) + self.images[0].shape
        assert_equal(array.shape, expected_shape)

    def test_concatenate_mismatched_image_shapes(self):
        with testing.raises(ValueError):
            self.images.concatenate()

    def test_multiimage_imagecollection(self):
        assert_equal(self.images_matched[0], self.frames_matched[0])
        assert_equal(self.images_matched[1], self.frames_matched[1])
