from skimage._shared.utils import copy_func
import numpy.testing as npt


def test_copyfunc():
    def foo(a):
        return a

    bar = copy_func(foo, name='bar')
    other = copy_func(foo)

    npt.assert_equal(bar.__name__, 'bar')
    npt.assert_equal(other.__name__, 'foo')

    other.__name__ = 'other'

    npt.assert_equal(foo.__name__, 'foo')


if __name__ == "__main__":
    npt.run_module_suite()
