"""Package version.

Ensure that this module remains standalone: i.e, no imports, or import *only* from the standard library.

Meson loads it as part of the build before ``_skimage2`` becomes importable.
"""

# The canonical library version is defined here.
# It is used by the meson build, {skimage,skimage2}.__version__, and docs
__version__ = '0.27.0rc2.dev0'


def _try_append_commit_info(version):
    """Append last commit date and hash to `version`, if available."""
    import subprocess
    from pathlib import Path

    try:
        output = subprocess.check_output(
            ['git', 'log', '-1', '--format="%H %aI"'],
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
            version += f'+git{git_date}.{git_hash[:7]}'

    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    except OSError:
        pass  # If skimage is built with emscripten which does not support processes

    return version


if 'dev' in __version__:
    __version__ = _try_append_commit_info(__version__)
