import os
import shutil
import sys

import click
from spin.cmds import meson
from spin import util


@click.command()
@click.option(
    "--clean", is_flag=True,
    default=False,
    help="Clean previously built docs before building"
)
@click.option(
    "--install-deps/--no-install-deps",
    default=True,
    help="Install dependencies before building"
)
def docs(clean, install_deps):
    """📖 Build documentation

    By default, SPHINXOPTS="-W", raising errors on warnings.
    To build without raising on warnings:

      SPHINXOPTS="" spin docs

    """
    if clean:
        doc_dir = "./doc/build"
        if os.path.isdir(doc_dir):
            print(f"Removing `{doc_dir}`")
            shutil.rmtree(doc_dir)

    site_path = meson._get_site_packages()
    if site_path is None:
        print("No built scikit-image found; run `spin build` first.")
        sys.exit(1)

    if install_deps:
        util.run(['pip', 'install', '-q', '-r', 'requirements/docs.txt'])

    os.environ['SPHINXOPTS'] = os.environ.get('SPHINXOPTS', "-W")

    os.environ['PYTHONPATH'] = f'{site_path}{os.sep}:{os.environ.get("PYTHONPATH", "")}'
    util.run(['make', '-C', 'doc', 'html'], replace=True)


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
def coverage():
    """📊 Generate coverage report
    """
    util.run(['python', '-m', 'spin', 'test', '--', '-o', 'python_functions=test_*', 'skimage', '--cov=skimage'], replace=True)


@click.command()
def sdist():
    """📦 Build a source distribution in `dist/`.
    """
    util.run(['python', '-m', 'build', '.', '--sdist'])
