from skimage._shared.utils import (copy_func, assert_nD, expand_arg)
import numpy.testing as npt
import numpy as np
from skimage._shared import testing


def test_assert_nD():
    z = np.random.random(200**2).reshape((200, 200))
    x = z[10:30, 30:10]
    with testing.raises(ValueError):
        assert_nD(x, 2)


def test_copyfunc():
    def foo(a):
        return a

    bar = copy_func(foo, name='bar')
    other = copy_func(foo)

    npt.assert_equal(bar.__name__, 'bar')
    npt.assert_equal(other.__name__, 'foo')

    other.__name__ = 'other'

    npt.assert_equal(foo.__name__, 'foo')


def test_expand_arg():
    out, rpl = expand_arg(None, 3)
    npt.assert_array_equal(out, np.zeros(3))
    assert rpl == 3

    out, rpl = expand_arg((None, 0, None, 42), 5, default=180)
    npt.assert_array_equal(out, np.array([180, 0, 180, 42, 180]))
    assert rpl == 1

    out, rpl = expand_arg((2.6, 2.3), 2, dtype=int)
    npt.assert_array_equal(out, np.array([2, 2], dtype=int))
    assert rpl == 0


if __name__ == "__main__":
    npt.run_module_suite()
