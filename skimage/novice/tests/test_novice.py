import os
import tempfile

import numpy as np
from numpy.testing import assert_equal, assert_raises, raises
from skimage import novice
from skimage import data_dir


IMAGE_PATH = os.path.join(data_dir, "elephant.png")
SMALL_IMAGE_PATH = os.path.join(data_dir, "block.png")


def test_pic_info():
    pic = novice.open(IMAGE_PATH)
    assert_equal(pic.format, "png")
    assert_equal(pic.path, os.path.abspath(IMAGE_PATH))
    assert_equal(pic.size, (665, 500))
    assert_equal(pic.width, 665)
    assert_equal(pic.height, 500)
    assert not pic.modified
    assert_equal(pic.scale, 1)


def test_pixel_iteration():
    pic = novice.open(SMALL_IMAGE_PATH)
    num_pixels = sum(1 for p in pic)
    assert_equal(num_pixels, pic.width * pic.height)


def test_modify():
    pic = novice.open(SMALL_IMAGE_PATH)
    assert_equal(pic.modified, False)

    for p in pic:
        if p.x < (pic.width / 2):
            p.red /= 2
            p.green /= 2
            p.blue /= 2

    for p in pic:
        if p.x < (pic.width / 2):
            assert p.red <= 128
            assert p.green <= 128
            assert p.blue <= 128

    s = pic.size
    pic.size = (pic.width / 2, pic.height / 2)
    assert_equal(pic.size, (int(s[0] / 2), int(s[1] / 2)))

    assert pic.modified
    assert pic.path is None


def test_pixel_rgb():
    pic = novice.Picture.from_size((3, 3), color=(10, 10, 10))
    pixel = pic[0, 0]
    pixel.rgb = range(3)

    assert_equal(pixel.rgb, range(3))
    for i, channel in enumerate((pixel.red, pixel.green, pixel.blue)):
        assert_equal(channel, i)

    pixel.red = 3
    pixel.green = 4
    pixel.blue = 5
    assert_equal(pixel.rgb, np.arange(3) + 3)

    for i, channel in enumerate((pixel.red, pixel.green, pixel.blue)):
        assert_equal(channel, i + 3)


def test_pixel_rgb_float():
    pixel = novice.Picture.from_size((1, 1))[0, 0]
    pixel.rgb = (1.1, 1.1, 1.1)
    assert_equal(pixel.rgb, (1, 1, 1))


@raises(ValueError)
def test_pixel_rgb_raises():
    pixel = novice.Picture.from_size((1, 1))[0, 0]
    pixel.rgb = (-1, -1, -1)


@raises(ValueError)
def test_pixel_red_raises():
    pixel = novice.Picture.from_size((1, 1))[0, 0]
    pixel.red = 256


@raises(ValueError)
def test_pixel_green_raises():
    pixel = novice.Picture.from_size((1, 1))[0, 0]
    pixel.green = 256


@raises(ValueError)
def test_pixel_blue_raises():
    pixel = novice.Picture.from_size((1, 1))[0, 0]
    pixel.blue = 256


def test_modified_on_set():
    pic = novice.Picture(SMALL_IMAGE_PATH)
    pic[0, 0] = (1, 1, 1)
    assert pic.modified
    assert pic.path is None


def test_modified_on_set_pixel():
    data = np.zeros(shape=(10, 5, 3), dtype=np.uint8)
    pic = novice.Picture(array=data)

    pixel = pic[0, 0]
    pixel.green = 1
    assert pic.modified


def test_update_on_save():
    pic = novice.Picture(array=np.zeros((3, 3)))
    pic.size = (6, 6)
    assert pic.modified
    assert pic.path is None

    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        pic.save(tmp.name)

        assert not pic.modified
        assert_equal(pic.path, os.path.abspath(tmp.name))
        assert_equal(pic.format, "jpeg")


def test_indexing():
    array = 128 * np.ones((10, 10, 3), dtype=np.uint8)
    pic = novice.Picture(array=array)

    pic[0:5, 0:5] = (0, 0, 0)
    for p in pic:
        if (p.x < 5) and (p.y < 5):
            assert_equal(p.rgb, (0, 0, 0))
            assert_equal(p.red, 0)
            assert_equal(p.green, 0)
            assert_equal(p.blue, 0)

    pic[:5, :5] = (255, 255, 255)
    for p in pic:
        if (p.x < 5) and (p.y < 5):
            assert_equal(p.rgb, (255, 255, 255))
            assert_equal(p.red, 255)
            assert_equal(p.green, 255)
            assert_equal(p.blue, 255)

    pic[5:pic.width, 5:pic.height] = (255, 0, 255)
    for p in pic:
        if (p.x >= 5) and (p.y >= 5):
            assert_equal(p.rgb, (255, 0, 255))
            assert_equal(p.red, 255)
            assert_equal(p.green, 0)
            assert_equal(p.blue, 255)

    pic[5:, 5:] = (0, 0, 255)
    for p in pic:
        if (p.x >= 5) and (p.y >= 5):
            assert_equal(p.rgb, (0, 0, 255))
            assert_equal(p.red, 0)
            assert_equal(p.green, 0)
            assert_equal(p.blue, 255)


def test_indexing_bounds():
    pic = novice.open(SMALL_IMAGE_PATH)

    # Outside bounds
    assert_raises(IndexError, lambda: pic[pic.width, pic.height])

    # Negative indexing not supported
    assert_raises(IndexError, lambda: pic[-1, -1])
    assert_raises(IndexError, lambda: pic[-1:, -1:])

    # Step sizes > 1 not supported
    assert_raises(IndexError, lambda: pic[::2, ::2])


def test_slicing():
    cut = 40
    pic = novice.open(IMAGE_PATH)
    rest = pic.width - cut
    temp = pic[:cut, :]
    pic[:rest, :] = pic[cut:, :]
    pic[rest:, :] = temp

    pic_orig = novice.open(IMAGE_PATH)

    # Check center line
    half_height = int(pic.height/2)
    for p1 in pic_orig[rest:, half_height]:
        for p2 in pic[:cut, half_height]:
            assert p1.rgb == p2.rgb

    for p1 in pic_orig[:cut, half_height]:
        for p2 in pic[rest:, half_height]:
            assert p1.rgb == p2.rgb


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
