import os
import sys

import click
import spin


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


@click.option(
    "--detect-dependencies",
    is_flag=True,
    default=None,
    help="Look for changes against main, "
    "detect which modules were involved, "
    "and test them and their dependencies. "
    "This overrides any pytest arguments.",
)
@spin.util.extend_command(spin.cmds.meson.test)
def test(*, parent_callback, detect_dependencies=False, **kwargs):
    if detect_dependencies:
        sys.path.insert(0, 'tools/')
        import module_dependencies

        p = spin.util.run(
            ['git', 'diff', 'main', '--stat', '--name-only'], output=False, echo=False
        )
        if p.returncode != 0:
            raise (click.ClickException('Could not git-diff against main'))

        git_diff = p.stdout.decode('utf-8')
        changed_modules = {
            mod
            for mod in module_dependencies._pkg_modules()
            if mod.replace('.', '/') in git_diff
        }

        pytest_args = kwargs.get('pytest_args', ())
        kwargs['pytest_args'] = (
            pytest_args
            + ('--pyargs',)
            + tuple(module_dependencies.modules_dependent_on(changed_modules))
        )

    parent_callback(**kwargs)
