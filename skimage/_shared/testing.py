"""Testing utilities."""


import os
import re
from tempfile import NamedTemporaryFile

from numpy import testing
import numpy as np
from skimage._shared._warnings import expected_warnings
import warnings

from .. import data, io, img_as_uint, img_as_float, img_as_int, img_as_ubyte


SKIP_RE = re.compile("(\s*>>>.*?)(\s*)#\s*skip\s+if\s+(.*)$")


def _assert_less(a, b, msg=None):
    message = "%r is not lower than %r" % (a, b)
    if msg is not None:
        message += ": " + msg
    assert a < b, message


def _assert_greater(a, b, msg=None):
    message = "%r is not greater than %r" % (a, b)
    if msg is not None:
        message += ": " + msg
    assert a > b, message


try:
    from nose.tools import assert_less
except ImportError:
    assert_less = _assert_less

try:
    from nose.tools import assert_greater
except ImportError:
    assert_greater = _assert_greater


def doctest_skip_parser(func):
    """ Decorator replaces custom skip test markup in doctests

    Say a function has a docstring::

        >>> something # skip if not HAVE_AMODULE
        >>> something + else
        >>> something # skip if HAVE_BMODULE

    This decorator will evaluate the expression after ``skip if``.  If this
    evaluates to True, then the comment is replaced by ``# doctest: +SKIP``.  If
    False, then the comment is just removed. The expression is evaluated in the
    ``globals`` scope of `func`.

    For example, if the module global ``HAVE_AMODULE`` is False, and module
    global ``HAVE_BMODULE`` is False, the returned function will have docstring::

        >>> something # doctest: +SKIP
        >>> something + else
        >>> something

    """
    lines = func.__doc__.split('\n')
    new_lines = []
    for line in lines:
        match = SKIP_RE.match(line)
        if match is None:
            new_lines.append(line)
            continue
        code, space, expr = match.groups()

        try:
            # Works as a function decorator
            if eval(expr, func.__globals__):
                code = code + space + "# doctest: +SKIP"
        except AttributeError:
            # Works as a class decorator
            if eval(expr, func.__init__.__globals__):
                code = code + space + "# doctest: +SKIP"

        new_lines.append(code)
    func.__doc__ = "\n".join(new_lines)
    return func


def roundtrip(img, plugin, suffix):
    """Save and read an image using a specified plugin"""
    if not '.' in suffix:
        suffix = '.' + suffix
    temp_file = NamedTemporaryFile(suffix=suffix, delete=False)
    temp_file.close()
    fname = temp_file.name
    io.imsave(fname, img, plugin=plugin)
    new = io.imread(fname, plugin=plugin)
    try:
        os.remove(fname)
    except Exception:
        pass
    return new


def color_check(plugin, fmt='png'):
    """Check roundtrip behavior for color images.

    All major input types should be handled as ubytes and read
    back correctly.
    """
    img = img_as_ubyte(data.chelsea())
    r1 = roundtrip(img, plugin, fmt)
    testing.assert_allclose(img, r1)

    img2 = img > 128
    r2 = roundtrip(img2, plugin, fmt)
    testing.assert_allclose(img2.astype(np.uint8), r2)

    img3 = img_as_float(img)
    with expected_warnings(['precision loss|unclosed file']):
        r3 = roundtrip(img3, plugin, fmt)
    testing.assert_allclose(r3, img)

    with expected_warnings(['precision loss']):
        img4 = img_as_int(img)
    if fmt.lower() in (('tif', 'tiff')):
        img4 -= 100
        with expected_warnings(['sign loss']):
            r4 = roundtrip(img4, plugin, fmt)
        testing.assert_allclose(r4, img4)
    else:
        with expected_warnings(['sign loss|precision loss|unclosed file']):
            r4 = roundtrip(img4, plugin, fmt)
            testing.assert_allclose(r4, img_as_ubyte(img4))

    img5 = img_as_uint(img)
    with expected_warnings(['precision loss|unclosed file']):
        r5 = roundtrip(img5, plugin, fmt)
    testing.assert_allclose(r5, img)


def mono_check(plugin, fmt='png'):
    """Check the roundtrip behavior for images that support most types.

    All major input types should be handled.
    """

    img = img_as_ubyte(data.moon())
    r1 = roundtrip(img, plugin, fmt)
    testing.assert_allclose(img, r1)

    img2 = img > 128
    r2 = roundtrip(img2, plugin, fmt)
    testing.assert_allclose(img2.astype(np.uint8), r2)

    img3 = img_as_float(img)
    with expected_warnings(['precision|unclosed file|\A\Z']):
        r3 = roundtrip(img3, plugin, fmt)
    if r3.dtype.kind == 'f':
        testing.assert_allclose(img3, r3)
    else:
        testing.assert_allclose(r3, img_as_uint(img))

    with expected_warnings(['precision loss|unclosed file']):
        img4 = img_as_int(img)
    if fmt.lower() in (('tif', 'tiff')):
        img4 -= 100
        with expected_warnings(['sign loss|\A\Z']):
            r4 = roundtrip(img4, plugin, fmt)
        testing.assert_allclose(r4, img4)
    else:
        with expected_warnings(['precision loss|sign loss|unclosed file']):
            r4 = roundtrip(img4, plugin, fmt)
            testing.assert_allclose(r4, img_as_uint(img4))

    img5 = img_as_uint(img)
    r5 = roundtrip(img5, plugin, fmt)
    testing.assert_allclose(r5, img5)


def setup_test():
    """Default package level setup routine for skimage tests.

    Import packages known to raise errors, and then
    force warnings to raise errors.
    Set a random seed
    """
    warnings.simplefilter('default')
    from scipy import signal, ndimage, special, optimize, linalg
    from scipy.io import loadmat
    from skimage import viewer, filter
    np.random.seed(0)
    warnings.simplefilter('error')


def teardown_test():
    """Default package level teardown routine for skimage tests.

    Restore warnings to default behavior
    """
    warnings.simplefilter('default')


if __name__ == '__main__':
    color_check('pil')
    mono_check('pil')
    mono_check('pil', 'bmp')
    mono_check('pil', 'tiff')
