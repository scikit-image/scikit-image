__all__ = ['modules_dependent_on', 'dependency_graph']


import numpy as np
import importlib
import multiprocessing
import functools
import sys
import os
import glob
import pytest


package = 'skimage'
included_mods = [
    # Add private and nested modules here
    '_shared',
    'filters.rank',
]
excluded_mods = [
    '__version__',
]


@functools.cache
def _pkg_modules() -> list[str]:
    """List all package submodules.

    Returns
    -------
    submodules : list
        Sorted list of package submodules.
    """
    pkg = importlib.import_module(package)

    members = {f'{package}.{attr}' for attr in dir(pkg)}
    included_mods_full = {f'{package}.{mod}' for mod in included_mods}
    excluded_mods_full = {f'{package}.{mod}' for mod in excluded_mods}

    submodules = sorted((members | included_mods_full) - excluded_mods_full)

    return submodules


def _import_dependencies(module: str) -> set[str]:
    # Check that process is fresh (we've had trouble with it being forked!)
    bad_modules = {mod for mod in sys.modules if f"{package}." in mod}
    if bad_modules:
        msg = f"Expected a fresh process, but {bad_modules} were already imported."
        raise RuntimeError(msg)

    # Import the module
    mod = importlib.import_module(module)

    # Import its test modules
    mod_dir = os.path.dirname(mod.__file__)
    test_files = glob.glob('**/test_*.py', root_dir=mod_dir, recursive=True)
    test_modules = [f"{module}/{f}".replace("/", ".")[:-3] for f in test_files]
    for mod in test_modules:
        try: importlib.import_module(mod)
        except pytest.skip.Exception: pass  # raised by `pytest.importorskip`

    # Return the modules that `module` depends on
    return set(_pkg_modules()) & set(sys.modules)


def dependency_graph():
    """Calculate the dependency graph of the current module.

    Each row represents the dependencies of one subpackage.
    `A[i, j]` is True if module `i` depends on (imports) module `j`.

    """
    mods = _pkg_modules()

    n = len(mods)
    A = np.zeros((n, n), dtype=bool)

    multiprocessing.set_start_method('spawn')

    with multiprocessing.Pool(maxtasksperchild=1) as p:
        mod_deps = p.imap(_import_dependencies, mods)
        for i, dependencies in enumerate(mod_deps):
            for mod in dependencies:
                j = mods.index(mod)
                A[i, j] = True

    return A


def dependency_toml():
    mods = _pkg_modules()
    mods_arr = np.array(mods, dtype=object)

    A = dependency_graph()

    toml = []
    for j, mod in enumerate(mods):
        dependency_names = mods_arr[A[:, j]].tolist()

        toml.append({"modules": {"path": mod, "depends_on": dependency_names}})

    return toml


def modules_dependent_on(modules: set[str] | list[str]) -> list[str]:
    """Return the set of modules that is a dependency of at least one of the given modules."""
    if not isinstance(modules, (set, list)):
        raise ValueError("`modules` must be set or list")

    pkg_mods = _pkg_modules()
    A = dependency_graph()
    A_j = np.zeros_like(A[:, 0])  # boolean indices of dependent modules
    for module in modules:
        j = pkg_mods.index(module)
        A_j |= A[:, j]

    return sorted(set(np.array(pkg_mods, dtype=object)[A_j]))


if __name__ == "__main__":
    toml = dependency_toml()
    for i, section in enumerate(toml):
        if i != 0:
            print()

        for key in section:
            print(f"[[{key}]]")
            for name, value in section[key].items():
                print(f"{name} = {repr(value)}")
