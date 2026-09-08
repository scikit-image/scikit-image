# Test doctest requirement machinery.

from _skimage2._shared.version_requirements import require


@require('foobar_from_skimage')
def doctest_require_but_fake():
    """>>> assert False"""
    # This doctest will fail unless doctest_requires set.


@require('scipy', version='<0.1')
def doctest_require_but_missing():
    """>>> assert False"""
    # This doctest will fail unless name + version set in doctest_requires.


@require('scipy', version='>0.1')
def doctest_require():
    """>>> assert True"""


@require('scipy')
def doctest_require_none():
    """>>> assert True"""
