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
    '_shared',
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

    # # Clear out sys.modules for when we spawn our import detector
    pkg_mods = [mod for mod in sys.modules if f"{package}." in mod]
    for mod in pkg_mods:
        del sys.modules[mod]

    return submodules


def _import_dependencies(module: str) -> set[str]:
    bad_sys_modules = {mod for mod in sys.modules if f"{package}." in mod}
    if bad_sys_modules:
        raise RuntimeError(
            f"Yikes! No package modules present at this point, but seeing {bad_sys_modules}!"
        )

    importlib.import_module(module)

    # This is a workaround for identifying the test modules
    # associated with an skimage submodule.
    #
    # Other libraries would need to do something different, and
    # we should think how to generalize.
    test_mod = importlib.import_module(f"{module}.tests")
    test_mod_dir = os.path.dirname(test_mod.__file__)
    test_files = [
        os.path.basename(f) for f in glob.glob(os.path.join(test_mod_dir, 'test_*.py'))
    ]
    test_modules = [f"{module}.tests.{os.path.splitext(f)[0]}" for f in test_files]
    for mod in test_modules:
        try:
            importlib.import_module(mod)
        except pytest.skip.Exception:  # raised by `pytest.importorskip`
            pass

    return set(_pkg_modules()) & set(sys.modules)


def dependency_graph():
    """Calculate the dependency graph of the current module.

    Each row represents the dependencies of one subpackage.

    """
    mods = _pkg_modules()

    n = len(mods)
    A = np.zeros((n, n), dtype=bool)

    with multiprocessing.Pool(maxtasksperchild=1) as p:
        mod_deps = p.map(_import_dependencies, mods)
        for k, dependencies in enumerate(mod_deps):
            for mod in dependencies:
                A[k, mods.index(mod)] = True

    return A


def dependency_toml():
    mods = _pkg_modules()
    mods_arr = np.array(mods, dtype=object)

    A = dependency_graph()

    toml = []
    for i, mod in enumerate(mods):
        dependency_names = mods_arr[A[:, i]].tolist()

        toml.append({"modules": {"path": mod, "depends_on": dependency_names}})

    return toml


def modules_dependent_on(modules: set[str] | list[str]) -> list[str]:
    """Return the set of modules that depend on any of the given modules."""
    if not isinstance(modules, (set, list)):
        raise ValueError("`modules` must be set or list")

    pkg_mods = _pkg_modules()
    A = dependency_graph()
    j = np.zeros_like(A[:, 0])  # boolean indices of dependent modules
    for module in modules:
        j |= A[:, pkg_mods.index(module)]

    return sorted(set(np.array(pkg_mods, dtype=object)[j]))


if __name__ == "__main__":
    toml = dependency_toml()
    for i, section in enumerate(toml):
        if i != 0:
            print()

        for key in section:
            print(f"[[{key}]]")
            for name, value in section[key].items():
                print(f"{name} = {repr(value)}")
