import os
import sys

import click
from spin.cmds import meson
from spin import util


@click.command()
@click.argument("sphinx_target", default="html")
@click.option(
    "--clean",
    is_flag=True,
    default=False,
    help="Clean previously built docs before building",
)
@click.option(
    "--build/--no-build",
    "first_build",
    default=True,
    help="Build project before generating docs",
)
@click.option(
    "--plot/--no-plot",
    "sphinx_gallery_plot",
    default=True,
    help="Sphinx gallery: enable/disable plots",
)
@click.option("--jobs", "-j", default="auto", help="Number of parallel build jobs")
@click.option(
    "--install-deps/--no-install-deps",
    default=False,
    help="Install dependencies before building",
)
@click.pass_context
def docs(
    ctx, sphinx_target, clean, first_build, jobs, sphinx_gallery_plot, install_deps
):
    """📖 Build documentation

    By default, SPHINXOPTS="-W", raising errors on warnings.
    To build without raising on warnings:

      SPHINXOPTS="" spin docs

    The command is roughly equivalent to `cd doc && make SPHINX_TARGET`.
    To get a list of viable `SPHINX_TARGET`:

      spin docs help

    """
    if install_deps:
        util.run(['pip', 'install', '-q', '-r', 'requirements/docs.txt'])

    for extra_param in ('install_deps',):
        del ctx.params[extra_param]
    ctx.forward(meson.docs)


@click.command()
@click.argument("asv_args", nargs=-1)
def asv(asv_args):
    """🏃 Run `asv` to collect benchmarks

    ASV_ARGS are passed through directly to asv, e.g.:

    spin asv -- dev -b TransformSuite

    Please see CONTRIBUTING.txt
    """
    site_path = meson._get_site_packages()
    if site_path is None:
        print("No built scikit-image found; run `spin build` first.")
        sys.exit(1)

    os.environ['PYTHONPATH'] = f'{site_path}{os.sep}:{os.environ.get("PYTHONPATH", "")}'
    util.run(['asv'] + list(asv_args))


@click.command()
def sdist():
    """📦 Build a source distribution in `dist/`."""
    util.run(['python', '-m', 'build', '.', '--sdist'])


@click.command(context_settings={'ignore_unknown_options': True})
@click.argument("ipython_args", metavar='', nargs=-1)
@click.pass_context
def ipython(ctx, ipython_args):
    """💻 Launch IPython shell with PYTHONPATH set

    OPTIONS are passed through directly to IPython, e.g.:

    spin ipython -i myscript.py
    """
    env = os.environ
    env['PYTHONWARNINGS'] = env.get('PYTHONWARNINGS', 'all')

    preimport = (
        r"import skimage as ski; "
        r"print(f'\nPreimported scikit-image {ski.__version__} as ski')"
    )
    ctx.params['ipython_args'] = (
        f"--TerminalIPythonApp.exec_lines={preimport}",
    ) + ipython_args

    ctx.forward(meson.ipython)
