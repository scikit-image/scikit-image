#!/usr/bin/env python
"""Pin asv.conf.json's numpy/scipy matrix to pyproject.toml's minimums.

Benchmarks then run against the versions the package promises to support,
instead of a fixed pin that drifts as that floor moves.
"""

import re
from pathlib import Path

import tomllib as toml

script_pth = Path(__file__)
repo_dir = script_pth.parent.parent


def min_numpy_scipy_versions(dependencies: list[str]) -> tuple[str, str]:
    """Return (numpy, scipy) minimums from [project.dependencies].

    Ignores the older [build-system] requires (pinned for build-time ABI
    compatibility) and the emscripten scipy floor.
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
