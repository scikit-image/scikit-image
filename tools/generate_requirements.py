#!/usr/bin/env python
"""Generate requirements/*.txt files from pyproject.toml."""

import sys
from pathlib import Path

try:  # standard module since Python 3.11
    import tomllib as toml
except ImportError:
    try:  # available for older Python via pip
        import tomli as toml
    except ImportError:
        sys.exit("Please install `tomli` first: `pip install tomli`")

repo_dir = (Path(__file__).parent / "..").resolve()
req_dir = repo_dir / "requirements"
pyproject = toml.loads((repo_dir / "pyproject.toml").read_text())

for key, opt_list in pyproject["project"]["optional-dependencies"].items():
    lines = ["# Generated from pyproject.toml"] + opt_list
    req_fname = req_dir / f"{key}.txt"
    req_fname.write_text("\n".join(lines) + "\n")
