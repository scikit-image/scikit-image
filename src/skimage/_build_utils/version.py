"""
Determine and print version number.

Used in top level ``meson.build``.

"""

from _skimage2._build_utils.version import (
    append_git_revision_and_date as append_git_revision_and_date,
    version_from_init as version_from_init,
)  # noqa: F401

__all__ = [
    'append_git_revision_and_date',
    'version_from_init',
]
