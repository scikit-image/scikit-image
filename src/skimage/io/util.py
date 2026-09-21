from _skimage2.io.util import (
    URL_REGEX as URL_REGEX,
    file_or_url_context as file_or_url_context,
    is_url as is_url,
)  # noqa: F401

__all__ = [
    'URL_REGEX',
    'file_or_url_context',
    'is_url',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
