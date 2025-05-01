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
    "--test-modified",
    is_flag=True,
    default=None,
    help="Test only modified submodules",
)
@click.option(
    "--test-modified-importers",
    is_flag=True,
    default=None,
    help="Test all submodules that import changed submodules.",
)
@spin.cmds.meson.build_dir_option
@spin.util.extend_command(spin.cmds.meson.test)
def test(
    *,
    parent_callback,
    build_dir,
    test_modified=False,
    test_modified_importers=False,
    **kwargs,
):
    if test_modified or test_modified_importers:
        sys.path.insert(0, 'tools/')
        import module_dependencies

        # Ensure spin-built version of skimage is accessible
        p = spin.cmds.meson._set_pythonpath(build_dir, quiet=True)
        sys.path.insert(0, p)

        pkg_mods = module_dependencies._pkg_modules()

        p = spin.util.run(
            ['git', 'diff', 'main', '--stat', '--name-only'], output=False, echo=False
        )
        if p.returncode != 0:
            raise (click.ClickException('Could not git-diff against main'))

        git_diff = p.stdout.decode('utf-8')
        changed_modules = {mod for mod in pkg_mods if mod.replace('.', '/') in git_diff}

        if test_modified:
            to_test = changed_modules
        else:
            to_test = set()

        if test_modified_importers:
            importers_of_modified = (
                set(module_dependencies.modules_dependent_on(changed_modules))
                - changed_modules
            )
            to_test = to_test | importers_of_modified

        pytest_args = kwargs.get('pytest_args', ())
        if "--pyargs" in pytest_args:
            raise RuntimeError(
                "--test-modified / --test-deps-of-modified will override --pyargs"
            )

        kwargs['pytest_args'] = pytest_args + ('--pyargs',) + tuple(sorted(to_test))

    kwargs['build_dir'] = build_dir
    parent_callback(**kwargs)
