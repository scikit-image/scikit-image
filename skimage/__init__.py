"""Image Processing for Python

scikit-image (a.k.a. ``skimage``) is a collection of algorithms for image
processing and computer vision.

The main package of ``skimage`` only provides a few utilities for converting
between image data types; for most features, you need to import one of its
subpackages.
"""

__version__ = '0.25.0rc2.dev0'

import lazy_loader as _lazy

__getattr__, __lazy_dir__, _ = _lazy.attach_stub(__name__, __file__)


def __dir__():
    """Add lazy-loaded attributes to keep consistent with `__getattr__`."""
    patched_dir = {*globals().keys(), *__lazy_dir__()}
    return sorted(patched_dir)


# `attach_stub` currently ignores __all__ inside the stub file and simply
# returns every lazy-imported object, so we need to define `__all__` again.
__all__ = [
    'color',
    'data',
    'draw',
    'exposure',
    'feature',
    'filters',
    'future',
    'graph',
    'io',
    'measure',
    'metrics',
    'morphology',
    'registration',
    'restoration',
    'segmentation',
    'transform',
    'util',
    '__version__',
]


# Logic for checking for improper install and importing while in the source
# tree when package has not been installed inplace.
# Code adapted from scikit-learn's __check_build module.
_INPLACE_MSG = """
It appears that you are importing a local scikit-image source tree. For
this, you need to have an inplace install. Maybe you are in the source
directory and you need to try from another location."""

_STANDARD_MSG = """
Your install of scikit-image appears to be broken.
Try re-installing the package following the instructions at:
https://scikit-image.org/docs/stable/user_guide/install.html"""


def _raise_build_error(e):
    # Raise a comprehensible error
    import os.path as osp

    local_dir = osp.split(__file__)[0]
    msg = _STANDARD_MSG
    if local_dir == "skimage":
        # Picking up the local install: this will work only if the
        # install is an 'inplace build'
        msg = _INPLACE_MSG
    raise ImportError(
        f"{e}\nIt seems that scikit-image has not been built correctly.\n{msg}"
    )


def _try_append_commit_info(version):
    """Append last commit date and hash to `version`, if available."""
    import subprocess
    from pathlib import Path

    try:
        output = subprocess.check_output(
            ['git', 'log', '-1', '--format="%h %aI"'],
            cwd=Path(__file__).parent,
            text=True,
        )
        if output:
            git_hash, git_date = (
                output.strip().replace('"', '').split('T')[0].replace('-', '').split()
            )
            version = '+'.join(
                [tag for tag in version.split('+') if not tag.startswith('git')]
            )
            version += f'+git{git_date}.{git_hash}'

    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    except OSError:
        pass  # If skimage is built with emscripten which does not support processes

    return version


if 'dev' in __version__:
    __version__ = _try_append_commit_info(__version__)


from skimage._shared.tester import PytestTester as _PytestTester

test = _PytestTester(__name__)
