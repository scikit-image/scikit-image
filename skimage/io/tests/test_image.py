from skimage.io import Image

from numpy.testing import assert_equal, assert_array_equal

def test_tags():
    f = Image([1, 2, 3], foo='bar', sigma='delta')
    g = Image([3, 2, 1], sun='moon')
    h = Image([1, 1, 1])

    assert_equal(f.tags['foo'], 'bar')
    assert_array_equal((g + 2).tags['sun'], 'moon')
    assert_equal(h.tags, {})

if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()

