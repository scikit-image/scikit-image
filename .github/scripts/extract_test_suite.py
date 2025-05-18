#!/usr/bin/python

"""Extract test suite of skimage into its own separate directory."""

import sys
import argparse
import traceback
import logging
import shutil
from contextlib import contextmanager
from pathlib import Path


logger = logging.getLogger(__name__)


def parse_command_line() -> dict:
    """Define and parse command line options."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("src", help="path to the source of scikit-image")
    parser.add_argument("dst", help="path to where the tests are extracted")
    parser.add_argument(
        "--debug", action="store_true", help="show debugging information"
    )
    kwargs = vars(parser.parse_args())
    return kwargs


@contextmanager
def handle_exceptions():
    """Handle (un)expected exceptions in `main()`."""
    try:
        yield
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


def walk_dir(root):
    """Walk each directory and file in the file tree of `root`.

    Iteration order is sorted alphabetically and the iteration descends into
    directories first.

    Parameters
    ----------
    root : Path

    Yields
    ------
    path : Path
        Each directory and file in `root`, including `root`.
    """
    yield root
    for path in sorted(root.iterdir()):
        if path.is_dir():
            yield from walk_dir(path)
        else:
            yield path


def main(src, dst, debug):
    """Execute script.

    Parameters
    ----------
    src : str
    dst : str
    debug : bool
    """
    src = Path(src)
    if not src.is_dir():
        raise ValueError(f"expected SRC to be a directory, got {src}")

    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=False)

    logger.info("copying conftest.py in root dir")
    shutil.copyfile(src / "conftest.py", dst / "conftest.py")
    logger.info("creating empty __init__.py in root dir")
    (dst / "__init__.py").touch()

    # Copy tests/ directories to dst, while emulating the sub dir structure
    for path in walk_dir(src):
        if path.is_dir() and str(path).endswith("tests"):
            target = dst / path.parent.relative_to(src)
            logger.info("copying %s to %s", path, target)
            shutil.copytree(path, target, dirs_exist_ok=True)


if __name__ == "__main__":
    with handle_exceptions():
        kwargs = parse_command_line()
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.DEBUG if kwargs["debug"] else logging.INFO,
            format="%(filename)s: %(message)s",
        )
        main(**kwargs)
