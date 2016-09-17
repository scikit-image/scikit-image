""" Testing decorators module
"""

import numpy as np
from nose.tools import (assert_true, assert_raises, assert_equal)
from skimage._shared.testing import doctest_skip_parser, test_parallel


def test_skipper():
    def f():
        pass

    class c():

        def __init__(self):
            self.me = "I think, therefore..."

    docstring = \
        """ Header

        >>> something # skip if not HAVE_AMODULE
        >>> something + else
        >>> a = 1 # skip if not HAVE_BMODULE
        >>> something2   # skip if HAVE_AMODULE
        """
    f.__doc__ = docstring
    c.__doc__ = docstring

    global HAVE_AMODULE, HAVE_BMODULE
    HAVE_AMODULE = False
    HAVE_BMODULE = True

    f2 = doctest_skip_parser(f)
    c2 = doctest_skip_parser(c)
    assert_true(f is f2)
    assert_true(c is c2)

    assert_equal(f2.__doc__,
                 """ Header

                 >>> something # doctest: +SKIP
                 >>> something + else
                 >>> a = 1
                 >>> something2
                 """)
    assert_equal(c2.__doc__,
                 """ Header

                 >>> something # doctest: +SKIP
                 >>> something + else
                 >>> a = 1
                 >>> something2
                 """)

    HAVE_AMODULE = True
    HAVE_BMODULE = False
    f.__doc__ = docstring
    c.__doc__ = docstring
    f2 = doctest_skip_parser(f)
    c2 = doctest_skip_parser(c)

    assert_true(f is f2)
    assert_equal(f2.__doc__,
                 """ Header

                 >>> something
                 >>> something + else
                 >>> a = 1 # doctest: +SKIP
                 >>> something2   # doctest: +SKIP
                 """)
    assert_equal(c2.__doc__,
                 """ Header

                 >>> something
                 >>> something + else
                 >>> a = 1 # doctest: +SKIP
                 >>> something2   # doctest: +SKIP
                 """)

    del HAVE_AMODULE
    f.__doc__ = docstring
    c.__doc__ = docstring
    assert_raises(NameError, doctest_skip_parser, f)
    assert_raises(NameError, doctest_skip_parser, c)


def test_test_parallel():
    state = []

    @test_parallel()
    def change_state1():
        state.append(None)
    change_state1()
    assert len(state) == 2

    @test_parallel(num_threads=1)
    def change_state2():
        state.append(None)
    change_state2()
    assert len(state) == 3

    @test_parallel(num_threads=3)
    def change_state3():
        state.append(None)
    change_state3()
    assert len(state) == 6


if __name__ == '__main__':
    np.testing.run_module_suite()
