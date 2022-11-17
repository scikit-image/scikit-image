import os
import shutil
import sys

import click
from devpy import util


@click.command()
@click.option(
    "--build-dir", default="build", help="Build directory; default is `$PWD/build`"
)
@click.option("--docs", is_flag=True, help="Remove only documentation build")
def clean(build_dir, docs=False):
    """🧹 Remove build directory.
    """
    if not docs:
        if os.path.isdir(build_dir):
            print(f"Removing `{build_dir}`")
            shutil.rmtree(build_dir)
        else:
            print(f"Build directory `{build_dir}` does not exist.")

    doc_dir = "./doc/build"
    if os.path.isdir(doc_dir):
        print(f"Removing `{doc_dir}`")
        shutil.rmtree(doc_dir)
    else:
        print(f"Documentation build `{doc_dir}` does not exist.")


@click.command()
@click.option(
    "--build-dir", default="build", help="Build directory; default is `$PWD/build`"
)
def docs(build_dir):
    """📖 Build documentation
    """
    site_path = util.get_site_packages(build_dir)
    if site_path is None:
        print("No built scikit-image found; run `./dev.py build` first.")
        sys.exit(1)

    util.run(['pip', 'install', '-q', '-r', 'requirements/docs.txt'])

    os.environ['SPHINXOPTS'] = '-W'
    os.environ['PYTHONPATH'] = f'{site_path}{os.sep}:{os.environ.get("PYTHONPATH", "")}'
    util.run(['make', '-C', 'doc', 'html'], replace=True)


@click.command()
@click.option(
    "--build-dir", default="build", help="Build directory; default is `$PWD/build`"
)
@click.argument("asv_args", nargs=-1)
def asv(build_dir, asv_args):
    """🏃 Run `asv` to collect benchmarks

    ASV_ARGS are passed through directly to asv, e.g.:

    ./dev.py asv -- dev -b TransformSuite

    Please see CONTRIBUTING.txt
    """
    site_path = util.get_site_packages(build_dir)
    if site_path is None:
        print("No built scikit-image found; run `./dev.py build` first.")
        sys.exit(1)

    os.environ['PYTHONPATH'] = f'{site_path}{os.sep}:{os.environ.get("PYTHONPATH", "")}'
    util.run(['asv'] + list(asv_args))


@click.command()
@click.option(
    "--build-dir", default="build", help="Build directory; default is `$PWD/build`"
)
def coverage(build_dir):
    """📊 Generate coverage report
    """
    util.run(['python', '-m', 'devpy', 'test', '--build-dir', build_dir, '--', '-o', 'python_functions=test_*', 'skimage', '--cov=skimage'], replace=True)
