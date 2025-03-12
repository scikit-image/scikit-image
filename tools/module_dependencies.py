__all__ = ['modules_dependent_on', 'dependency_graph']


import numpy as np
import importlib
import multiprocessing
import functools
import sys


package = 'skimage'
included_mods = [
    '_shared',
]
excluded_mods = [
    '__version__',
]


@functools.cache
def _pkg_modules() -> list[str]:
    pkg = importlib.import_module(package)

    members = {f'{package}.{attr}' for attr in dir(pkg)}
    included_mods_full = {f'{package}.{mod}' for mod in included_mods}
    excluded_mods_full = {f'{package}.{mod}' for mod in excluded_mods}

    return (members | set(included_mods_full)) - set(excluded_mods_full)


@functools.cache
def _pkg_modules_index():
    mods = sorted(_pkg_modules())
    return {mod: index for index, mod in enumerate(mods)}


def _import_dependencies(module: str | list[str]) -> set[str]:
    importlib.import_module(module)
    importlib.import_module(f"{module}.tests")

    pkg_modules = _pkg_modules()
    pkg_sys_modules = {mod for mod in sys.modules if package in mod}

    return pkg_modules & pkg_sys_modules


def dependency_graph():
    """Calculate the dependency graph of the current module.

    Each row represents the dependencies of one subpackage.

    """
    mods = sorted(_pkg_modules())  # sort for stable matrix
    mods_idx = _pkg_modules_index()

    n = len(mods)
    A = np.zeros((n, n), dtype=int)

    with multiprocessing.Pool() as p:
        mod_deps = p.map(_import_dependencies, mods)
        for k, dependencies in enumerate(mod_deps):
            for mod in dependencies:
                A[k, mods_idx[mod]] = 1

    return A


def modules_dependent_on(modules: set[str] | list[str]) -> set[str]:
    """Return the set of modules that depend on any of the given modules."""
    changed_modules = modules

    A = dependency_graph().astype(bool)

    pkg_mods = np.array(sorted(_pkg_modules_index()), dtype=object)
    pkg_mods_idx = _pkg_modules_index()

    all_dependent_mods = []
    for changed_mod in changed_modules:
        dependent_mods = pkg_mods[A[:, pkg_mods_idx[changed_mod]]]
        all_dependent_mods.extend(dependent_mods.tolist())

    return set(all_dependent_mods)
