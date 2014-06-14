__all__ = ['all_warnings']

from contextlib import contextmanager
import sys
import warnings
import inspect


@contextmanager
def all_warnings():
    """
    Context for use in testing to ensure that all warnings are raised.

    Examples
    --------
    >>> import warnings
    >>> def foo():
    ...     warnings.warn(RuntimeWarning("bar"))

    We raise the warning once, while the warning filter is set to "once".
    Hereafter, the warning is invisible, even with custom filters:

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter('once')
    ...     foo()

    We can now run ``foo()`` without a warning being raised:

    >>> from numpy.testing import assert_warns
    >>> foo()

    To catch the warning, we call in the help of ``all_warnings``:

    >>> with all_warnings():
    ...     assert_warns(RuntimeWarning, foo)
    """

    # Whenever a warning is triggered, Python adds a __warningregistry__
    # member to the *calling* module.  The exercize here is to find
    # and eradicate all those breadcrumbs that were left lying around.
    #
    # We proceed by first searching all parent calling frames and explicitly
    # clearing their warning registries (necessary for the doctests above to
    # pass).  Then, we search for all submodules of skimage and clear theirs
    # as well (necessary for the skimage test suite to pass).

    frame = inspect.currentframe()
    if frame:
        for f in inspect.getouterframes(frame):
            f[0].f_locals['__warningregistry__'] = {}
    del frame

    for mod_name, mod in list(sys.modules.items()):
        if 'six.moves' in mod_name:
            continue
        try:
            mod.__warningregistry__.clear()
        except AttributeError:
            pass

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yield w
