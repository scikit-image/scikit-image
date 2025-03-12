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
def _pkg_modules() -> tuple[list[str], dict[str, int]]:
    """List all package submodules.

    Returns
    -------
    submodules : list
        Sorted list of package submodules.
    submodule_idx : dict
        Mapping of submodule to integer, its index
        in `submodules`.
    """
    pkg = importlib.import_module(package)

    members = {f'{package}.{attr}' for attr in dir(pkg)}
    included_mods_full = {f'{package}.{mod}' for mod in included_mods}
    excluded_mods_full = {f'{package}.{mod}' for mod in excluded_mods}

    # Sort entries, so that all adjacency matrix calculations are stable
    submodules = sorted((members | included_mods_full) - excluded_mods_full)
    submodule_idx = {mod: index for index, mod in enumerate(submodules)}

    return submodules, submodule_idx


def _import_dependencies(module: str) -> set[str]:
    importlib.import_module(module)
    importlib.import_module(f"{module}.tests")

    pkg_modules, _ = _pkg_modules()
    pkg_sys_modules = {mod for mod in sys.modules if f"{package}." in mod}

    return set(pkg_modules) & pkg_sys_modules


def dependency_graph():
    """Calculate the dependency graph of the current module.

    Each row represents the dependencies of one subpackage.

    """
    mods, mods_idx = _pkg_modules()

    n = len(mods)
    A = np.zeros((n, n), dtype=bool)

    with multiprocessing.Pool() as p:
        mod_deps = p.map(_import_dependencies, mods)
        for k, dependencies in enumerate(mod_deps):
            for mod in dependencies:
                A[k, mods_idx[mod]] = True

    return A


def dependency_toml():
    mods, mods_idx = _pkg_modules()
    mods_arr = np.array(mods, dtype=object)

    A = dependency_graph()

    toml = []
    for i, mod in enumerate(mods):
        dependency_names = mods_arr[A[:, i]].tolist()

        toml.append({"modules": {"path": mod, "depends_on": dependency_names}})

    return toml


def modules_dependent_on(modules: set[str] | list[str]) -> list[str]:
    """Return the set of modules that depend on any of the given modules."""
    changed_modules = modules

    A = dependency_graph().astype(bool)

    pkg_mods, pkg_mods_idx = _pkg_modules()
    pkg_mods_arr: np.typing.NDArray = np.array(pkg_mods, dtype=object)

    all_dependent_mods = []
    for changed_mod in changed_modules:
        dependent_mods = pkg_mods_arr[A[:, pkg_mods_idx[changed_mod]]]
        all_dependent_mods.extend(dependent_mods.tolist())

    return sorted(set(all_dependent_mods))


if __name__ == "__main__":
    toml = dependency_toml()
    for i, section in enumerate(toml):
        if i != 0:
            print()

        for key in section:
            print(f"[[{key}]]")
            for name, value in section[key].items():
                print(f"{name} = {repr(value)}")
