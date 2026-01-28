import os
import sys

import click
import spin
from spin.cmds.meson import _is_editable_install_of_same_source


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


@click.option("--doctest/--no-doctest", default=True, help="Whether to run doctests.")
@spin.util.extend_command(spin.cmds.meson.test)
def test(*, parent_callback, doctest=False, **kwargs):
    pytest_args = kwargs.get('pytest_args', ())

    is_out_of_tree_build = not _is_editable_install_of_same_source("scikit-image")
    if is_out_of_tree_build and "src" in str(pytest_args):
        click.secho(_SRC_IN_TEST_ARGS_WARNING_MESSAGE, fg="yellow", bold=True)

    if doctest:
        if '--doctest-plus' not in pytest_args:
            pytest_args = ('--doctest-plus',) + pytest_args

    kwargs["pytest_args"] = pytest_args
    parent_callback(**kwargs)
