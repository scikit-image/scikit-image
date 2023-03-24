"""Custom Sphinx extensions for scikit-image's docs."""

import re

from sphinx.directives.other import TocTree


def natural_sort_key(item):
    """Transforms entries into tuples that can be sorted in natural order [1]_.

    This can be passed to the "key" argument of Python's `sorted` function.

    Parameters
    ----------
    item :
        Item to generate the key from. `str` is called on this item before generating
        the key.

    Returns
    -------
    key : tuple[str or int]
        Key to sort by.

    Examples
    --------
    >>> natural_sort_key("release_notes_2.rst")
    ('release_notes_', 2, '.rst')
    >>> natural_sort_key("release_notes_10.rst")
    ('release_notes_', 10, '.rst')
    >>> sorted(["10.b", "2.c", "100.a"], key=natural_sort_key)
    ['2.c', '10.b', '100.a']

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Natural_sort_order
    """
    splitted = re.split(r"(\d+)", str(item))
    key = tuple(int(x) if x.isdigit() else x for x in splitted)
    return key


class NaturalSortedTocTree(TocTree):
    """Directive that sorts all TOC entries in natural order by their file names.

    Behaves similar to Sphinx's default ``toctree`` directive. The ``reversed`` option
    is respected, though the given order of entries (or globbed entries) is ignored.
    """

    def parse_content(self, toctree):
        ret = super().parse_content(toctree)
        reverse = 'reversed' in self.options
        toctree['entries'] = sorted(
            toctree['entries'], key=natural_sort_key, reverse=reverse
        )
        return ret


def setup(app):
    app.add_directive('naturalsortedtoctree', NaturalSortedTocTree)
