import os
import subprocess
import sys

import click
import spin
from spin.cmds.meson import _is_editable_install_of_same_source

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@click.option(
    "--install-deps/--no-install-deps",
    default=False,
    help="Install dependencies before building",
)
@spin.util.extend_command(spin.cmds.meson.docs)
def docs(*, parent_callback, install_deps, **kwargs):
    if install_deps:
        spin.util.run(['pip', 'install', '-q', '-r', 'requirements/docs.txt'])

    parent_callback(**kwargs)


# Override default jobs to 1
jobs_param = next(p for p in docs.params if p.name == 'jobs')
jobs_param.default = 1


@click.command()
@click.argument("asv_args", nargs=-1)
@spin.cmds.meson.build_dir_option
def asv(asv_args, build_dir):
    """🏃 Run `asv` to collect benchmarks

    ASV_ARGS are passed through directly to asv, e.g.:

    spin asv -- dev -b TransformSuite

    Please see CONTRIBUTING.txt
    """
    site_path = spin.cmds.meson._get_site_packages(build_dir)
    if site_path is None:
        print("No built scikit-image found; run `spin build` first.")
        sys.exit(1)

    os.environ['PYTHONPATH'] = f'{site_path}{os.sep}:{os.environ.get("PYTHONPATH", "")}'
    spin.util.run(['asv'] + list(asv_args))


@spin.util.extend_command(spin.cmds.meson.ipython)
def ipython(*, parent_callback, **kwargs):
    env = os.environ
    env['PYTHONWARNINGS'] = env.get('PYTHONWARNINGS', 'all')

    pre_import = (
        r"import skimage as ski; "
        r"print(f'\nPreimported scikit-image {ski.__version__} as ski')"
    )
    parent_callback(pre_import=pre_import, **kwargs)


@click.command()
@click.argument("pyproject-build-args", metavar="", nargs=-1)
def sdist(pyproject_build_args):
    """📦 Build a source distribution in `dist/`

    Extra arguments are passed to `pyproject-build`, e.g.

      spin sdist -- -x -n
    """
    p = spin.util.run(
        ["pyproject-build", ".", "--sdist"] + list(pyproject_build_args), output=False
    )
    try:
        built_line = next(
            line
            for line in p.stdout.decode('utf-8').split('\n')
            if line.startswith('Successfully built')
        )
    except StopIteration:
        print("Error: could not identify built wheel")
        sys.exit(1)
    print(built_line)
    sdist = os.path.join('dist', built_line.replace('Successfully built ', ''))
    print(f"Validating {sdist}...")
    spin.util.run(["tools/check_sdist.py", sdist])


_SRC_IN_TEST_ARGS_WARNING_MESSAGE = """\
WARNING: Found 'src' in test arguments and using out-of-tree build.
For out-of-tree builds, selecting `src/` as a doctest path may fail because
Pytest doesn't expect source and installation to live in different places.
Use an editable install (`spin install`) which supports this or avoid passing
`src/`. For example:
    spin test -- src/skimage/io  # NO!
    spin test -- skimage.io      # YES!
"""


def _get_skimage_subpackages(build_dir=None):
    """Return the set of all skimage/skimage2/_skimage2 subpackage names.

    If *build_dir* is None, skimage is expected to already be importable
    (e.g. installed from a wheel via pip).
    """
    import importlib

    if build_dir is not None:
        p = spin.cmds.meson._set_pythonpath(build_dir, quiet=True)
        sys.path.insert(0, p)

    pkg = importlib.import_module('skimage')
    pkg_mods = {f'skimage.{attr}' for attr in dir(pkg) if not attr.startswith('_')}
    pkg_mods |= {'skimage._shared', 'skimage.filters.rank'}
    pkg_mods -= {'skimage.__version__'}
    # Include changes to skimage2 and _skimage2
    pkg_mods |= {mod.replace("skimage.", "skimage2.") for mod in pkg_mods}
    pkg_mods |= {mod.replace("skimage.", "_skimage2.") for mod in pkg_mods}
    return pkg_mods


def _get_changed_subpackages(base_ref, pkg_mods):
    """Return the set of changed subpackages relative to *base_ref*, with cross-package expansion."""
    base_ref = base_ref or os.environ.get('GITHUB_BASE_REF') or 'main'
    # In CI, the base branch is only available as origin/<base_ref> after fetch
    p = subprocess.run(
        ['git', 'rev-parse', '--verify', base_ref],
        capture_output=True,
        cwd=_REPO_ROOT,
    )
    if p.returncode != 0:
        base_ref = f'origin/{base_ref}'

    p = subprocess.run(
        ['git', 'merge-base', base_ref, 'HEAD'],
        capture_output=True,
        cwd=_REPO_ROOT,
    )
    if p.returncode != 0:
        # Shallow clone: fall back to diffing directly against the base ref.
        # For GitHub PR merge commits, base_ref is a direct parent of HEAD so
        # this is equivalent to the merge-base diff.
        merge_base = base_ref
    else:
        merge_base = p.stdout.decode('utf-8').strip()

    p = subprocess.run(
        ['git', 'diff', merge_base, '--name-only'],
        capture_output=True,
        cwd=_REPO_ROOT,
    )
    if p.returncode != 0:
        raise click.ClickException(f'Could not git-diff against {base_ref!r}')

    git_diff = p.stdout.decode('utf-8')
    changed_subpackages = {mod for mod in pkg_mods if mod.replace('.', '/') in git_diff}

    # Cross-package expansion:
    # - skimage.X or _skimage2.X changed → also test skimage2.X
    # - skimage2.X or _skimage2.X changed → also test skimage.X
    companions = {
        'skimage.': ['skimage2.'],
        'skimage2.': ['skimage.'],
        '_skimage2.': ['skimage.', 'skimage2.'],
    }
    expanded = set(changed_subpackages)
    for mod in changed_subpackages:
        for prefix, targets in companions.items():
            if mod.startswith(prefix):
                base = mod[len(prefix) :]
                expanded.update(t + base for t in targets if t + base in pkg_mods)
                break
    return expanded


def _get_test_paths(changed_subpackages, doctest, installed=False):
    """Map module names to their test and (optionally) source directories.

    src-layout: tests live outside src/, doctests live inside src/.
    _skimage2 tests live in tests/skimage2/ (not tests/_skimage2/)
    skimage2 doctests live in src/_skimage2/ (not src/skimage2/)

    When *installed* is True, doctest paths are returned as module names
    (e.g. ``skimage.metrics``) rather than ``src/`` filesystem paths, so
    that pytest with ``--pyargs`` collects doctests from the installed
    package in site-packages.  This avoids a ``warn_external`` mismatch
    where ``skip_file_prefixes`` points to site-packages paths but the
    call-stack frames point to the source tree.
    """
    test_paths = []
    seen = set()
    for mod in sorted(changed_subpackages):
        mod_path = mod.replace('.', '/')
        test_path = mod_path.replace('_skimage2/', 'skimage2/', 1)
        test_dir = os.path.join(_REPO_ROOT, 'tests', test_path)
        if test_dir not in seen and os.path.isdir(test_dir):
            test_paths.append(test_dir)
            seen.add(test_dir)
        if doctest:
            if installed:
                # Use the installed module name so --pyargs finds doctests in
                # site-packages.  skimage2.X is re-exported from _skimage2.X,
                # which is the actual installed package (see pyproject.toml).
                installed_mod = (
                    '_skimage2' + mod[len('skimage2') :]
                    if mod.startswith('skimage2.')
                    else mod
                )
                if installed_mod not in seen:
                    test_paths.append(installed_mod)
                    seen.add(installed_mod)
            else:
                src_path = mod_path.replace('skimage2/', '_skimage2/', 1)
                src_dir = os.path.join(_REPO_ROOT, 'src', src_path)
                if src_dir not in seen and os.path.isdir(src_dir):
                    test_paths.append(src_dir)
                    seen.add(src_dir)
    return test_paths


@click.option(
    "--installed",
    is_flag=True,
    default=False,
    help=(
        "Run tests against an installed package (e.g. a pip-installed wheel) "
        "instead of a spin/meson build directory. "
        "Pytest is invoked directly without any spin build-environment setup."
    ),
)
@click.option(
    "--test-modified",
    is_flag=True,
    default=False,
    help="Test only modified subpackages",
)
@click.option("--doctest/--no-doctest", default=True, help="Whether to run doctests.")
@click.option(
    "--base-ref",
    default=None,
    help=(
        "Base ref for detecting modified subpackages (default: $GITHUB_BASE_REF or 'main')"
    ),
)
@spin.util.extend_command(spin.cmds.meson.test)
def test(
    *,
    parent_callback,
    installed=False,
    test_modified=False,
    doctest=False,
    base_ref=None,
    **kwargs,
):
    pytest_args = kwargs.get('pytest_args', ())

    # For out-of-tree builds, src/ paths don't work as doctest collection paths.
    # Short-circuits when installed=True so _is_editable_install_of_same_source is
    # never called in that mode.
    is_out_of_tree_build = not installed and not _is_editable_install_of_same_source(
        "scikit-image"
    )

    if test_modified:
        if "--pyargs" in pytest_args:
            raise RuntimeError("--test-modified will override --pyargs")

        build_dir = None if installed else kwargs.get('build_dir', 'build')
        pkg_mods = _get_skimage_subpackages(build_dir)
        changed_subpackages = _get_changed_subpackages(base_ref, pkg_mods)

        if not changed_subpackages:
            click.secho("No modified skimage subpackages detected.", fg="yellow")
            return

        click.secho(
            f"Testing modified subpackages: {', '.join(sorted(changed_subpackages))}",
            fg="green",
        )

        # src/ paths are only valid for doctests in installed/editable mode;
        # out-of-tree builds must skip them to avoid silent collection failures.
        test_paths = _get_test_paths(
            changed_subpackages, doctest and not is_out_of_tree_build, installed
        )
        pytest_args = pytest_args + tuple(test_paths)

    if is_out_of_tree_build and "src" in str(pytest_args):
        click.secho(_SRC_IN_TEST_ARGS_WARNING_MESSAGE, fg="yellow", bold=True)

    if doctest:
        if '--doctest-plus' not in pytest_args:
            pytest_args = ('--doctest-plus',) + pytest_args

    if installed:
        spin.util.run(['pytest'] + list(pytest_args))
    else:
        kwargs["pytest_args"] = pytest_args
        parent_callback(**kwargs)
