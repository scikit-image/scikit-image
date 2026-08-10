#!/usr/bin/env python
"""Pin asv.conf.json's numpy/scipy matrix to pyproject.toml's minimums.

Keeps the benchmarked environment in sync with the versions the
package actually promises to support, rather than an arbitrary fixed
pin that can silently drift as pyproject.toml's floor moves.
"""

import re
from pathlib import Path

import tomllib as toml

script_pth = Path(__file__)
repo_dir = script_pth.parent.parent


def min_numpy_scipy_versions(dependencies: list[str]) -> tuple[str, str]:
    """Return (numpy, scipy) minimums from [project.dependencies].

    Deliberately reads [project.dependencies], not [build-system] ->
    requires, which pins older build-time versions for ABI
    compatibility rather than the runtime support floor. Uses the
    non-emscripten scipy floor since benchmarks don't run on Pyodide.
    """
    numpy_version = next(
        re.match(r"numpy>=([0-9.]+)", dep).group(1)
        for dep in dependencies
        if dep.startswith("numpy>=")
    )
    scipy_version = next(
        re.match(r"scipy>=([0-9.]+)", dep).group(1)
        for dep in dependencies
        if dep.startswith("scipy>=") and '!= "emscripten"' in dep
    )
    return numpy_version, scipy_version


def main() -> None:
    pyproject = toml.loads((repo_dir / "pyproject.toml").read_text())
    numpy_version, scipy_version = min_numpy_scipy_versions(
        pyproject["project"]["dependencies"]
    )

    conf_path = repo_dir / "asv.conf.json"
    text = conf_path.read_text()
    text = re.sub(r'"numpy": \[[^\]]*\]', f'"numpy": ["{numpy_version}"]', text)
    text = re.sub(r'"scipy": \[[^\]]*\]', f'"scipy": ["{scipy_version}"]', text)
    conf_path.write_text(text)


if __name__ == "__main__":
    main()
