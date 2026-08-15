"""Read the benchmark run configuration beside this file.

Shared by CI (.github/scripts/resolve-benchmark-params.py) and local
tooling (`spin asv`, see .spin/_asv.py) so the two scope and configure
a run identically. Kept to the standard library, and free of imports
from the benchmarks package itself, so both can load it by path on a
runner with no numpy or skimage installed.
"""

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES_FILE = os.path.join(_HERE, "profiles.json")
MODULE_MAP_FILE = os.path.join(_HERE, "module-map.json")
ASV_CONF_FILE = os.path.join(os.path.dirname(_HERE), "asv.conf.json")


def _read(path: str) -> dict:
    """A config file, without the keys documenting it."""
    with open(path) as f:
        entries = json.load(f)
    return {k: v for k, v in entries.items() if not k.startswith("_")}


def asv_pythons() -> list:
    """The Python versions asv.conf.json declares support for.

    The single source of truth for the version CI builds against and
    the one `spin asv` expects to be running under.
    """
    with open(ASV_CONF_FILE) as f:
        return json.load(f)["pythons"]


def profile_names() -> list:
    """Every profile profiles.json defines."""
    return sorted(_read(PROFILES_FILE))


def load_profile(name: str) -> dict:
    """The asv settings a profile in profiles.json names.

    Raises KeyError if there's no such profile.
    """
    profile = _read(PROFILES_FILE)[name]
    return {k: v for k, v in profile.items() if k != "description"}


def modules_for_paths(paths) -> list:
    """The benchmark modules covering the given repository paths.

    Paths under a subpackage that module-map.json doesn't list, or
    outside src/skimage/ entirely, contribute nothing.
    """
    modules = []
    for pkg, pkg_modules in _read(MODULE_MAP_FILE).items():
        if any(path.startswith(f"src/skimage/{pkg}/") for path in paths):
            modules.extend(pkg_modules)
    return modules


def bench_filter(paths) -> str:
    """An asv -b regex covering those paths, or empty if none match."""
    modules = modules_for_paths(paths)
    return f"^({'|'.join(modules)})\\." if modules else ""
