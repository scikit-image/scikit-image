import os
import tempfile

from numpy.testing import assert_equal, assert_raises
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
    assert_equal(pic.modified, False)
    assert_equal(pic.inflation, 1)

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
            assert_equal(p.red <= 128, True)
            assert_equal(p.green <= 128, True)
            assert_equal(p.blue <= 128, True)

    s = pic.size
    pic.size = (pic.width / 2, pic.height / 2)
    assert_equal(pic.size, (int(s[0] / 2), int(s[1] / 2)))

    assert_equal(pic.modified, True)
    assert_equal(pic.path, None)

    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        pic.save(tmp.name)

        assert_equal(pic.modified, False)
        assert_equal(pic.path, os.path.abspath(tmp.name))
        assert_equal(pic.format, "jpeg")


def test_indexing():
    pic = novice.open(SMALL_IMAGE_PATH)

    # Slicing
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
