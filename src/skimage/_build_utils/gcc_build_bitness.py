"""
Detect bitness (32 or 64) of Mingw-w64 gcc build target on Windows.
"""

from _skimage2._build_utils.gcc_build_bitness import main as main  # noqa: F401

__all__ = ['main']
