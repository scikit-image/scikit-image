"""Benchmark commands for `spin`.

Wraps `asv` so a local run can be scoped and configured the way CI
scopes and configures its own, through the benchmarks/config.py that
.github/scripts/resolve-benchmark-params.py reads as well.
"""

import importlib.util
import os
import subprocess
import sys

import click
import spin

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Loaded by path rather than imported: benchmarks/__init__.py pulls in
# numpy and skimage, which reading configuration shouldn't require.
_spec = importlib.util.spec_from_file_location(
    "benchmark_config", os.path.join(_REPO_ROOT, "benchmarks", "config.py")
)
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)


# asv subcommands that build an environment from asv.conf.json. The
# rest either read stored results (compare, show) or don't touch
# environments at all (machine, publish, preview, ...).
_ASV_ENV_BUILDING = {"check", "continuous", "find", "profile", "run", "setup"}


def _asv_builds_declared_environment(asv_args):
    """Whether this asv invocation builds an environment matching
    asv.conf.json's declared `pythons`.

    It doesn't when the subcommand never builds one, or when
    `-E existing`/`--python=same` points asv at the interpreter already
    running, which works with any version and so shouldn't be gated on
    matching asv.conf.json.
    """
    if not asv_args or asv_args[0] not in _ASV_ENV_BUILDING:
        return False
    return not any(
        "existing" in arg or arg == "same" or arg.endswith((":same", "=same"))
        for arg in asv_args
    )


def _changed_bench_filter(base):
    """An asv `-b` regex covering the subpackages changed since base.

    Uses the merge base so unrelated commits on base don't count, and
    diffs against the working tree so uncommitted edits do. Returns
    None when nothing benchmarked changed.
    """
    merge_base = subprocess.run(
        ["git", "merge-base", base, "HEAD"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()
    changed = subprocess.run(
        ["git", "diff", "--name-only", merge_base, "--", "src/skimage/"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.splitlines()
    return _config.bench_filter(changed) or None


def _apply_asv_profile(name, asv_args):
    """Set up asv to measure what CI's named profile measures.

    ASV_SKIP_SLOW and ASV_FULL_PARAMS are read by benchmarks/__init__.py
    and go into the environment. The comparison factor and process count
    are asv's own flags, so they're only added where asv accepts them,
    and never over an equivalent flag already given explicitly.
    """
    try:
        profile = _config.load_profile(name)
    except KeyError:
        raise click.BadParameter(
            f"unknown profile {name!r}; choose from {_config.profile_names()}"
        )

    os.environ["ASV_SKIP_SLOW"] = profile["ASV_SKIP_SLOW"]
    os.environ["ASV_FULL_PARAMS"] = profile["ASV_FULL_PARAMS"]

    args = list(asv_args)
    if not args:
        return args
    subcommand, rest = args[0], args[1:]

    injected = []
    if subcommand == "continuous" and not any(a.startswith("--factor") for a in rest):
        injected += ["--factor", profile["ASV_FACTOR"]]
    if subcommand in {"continuous", "run"} and not any(
        a == "-a" or a.startswith("--attribute") for a in rest
    ):
        injected += ["-a", f"processes={profile['ASV_PROCESSES']}"]

    return [subcommand] + injected + rest


def _check_asv_python_version():
    """Fail fast (with a clear message) if the active Python isn't one
    of the versions asv.conf.json declares support for - asv itself
    would otherwise fail deep inside env creation with a much more
    cryptic "no such python" error.
    """
    pythons = _config.asv_pythons()

    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    if current not in pythons:
        print(
            f"Active Python is {current}, but asv.conf.json only declares "
            f"support for {pythons}. Rebuild with one of those versions "
            "before running `spin asv`."
        )
        sys.exit(1)


@click.command()
@click.argument("asv_args", nargs=-1)
@click.option(
    "--profile",
    type=click.Choice(_config.profile_names()),
    help="Measure what a 'fast' (trimmed, as pull request checks run) or "
    "'full' (nightly) run measures (see benchmarks/profiles.json).",
)
@click.option(
    "--changed",
    is_flag=True,
    help="Only run the benchmarks covering subpackages you've changed, "
    "the way a pull request check scopes itself.",
)
@click.option(
    "--changed-base",
    default="main",
    show_default=True,
    help="Ref that --changed compares against.",
)
@spin.cmds.meson.build_dir_option
def asv(asv_args, profile, changed, changed_base, build_dir):
    """🏃 Run `asv` to collect benchmarks

    ASV_ARGS are passed through directly to asv, e.g.:

    spin asv -- check -v -E existing

    Pass --profile to match a CI run's benchmark selection and
    comparison factor, and --changed to scope the run to the
    subpackages you've touched, as a pull request check does:

    spin asv --profile fast --changed -- continuous main

    Please see CONTRIBUTING.txt
    """
    if profile:
        asv_args = _apply_asv_profile(profile, asv_args)

    if changed:
        if any(a == "-b" or a.startswith("--bench") for a in asv_args):
            print("--changed was ignored: an explicit -b/--bench filter is set.")
        else:
            bench_filter = _changed_bench_filter(changed_base)
            if bench_filter is None:
                print(
                    f"No benchmarked subpackage changed since {changed_base}; "
                    "nothing to run."
                )
                return
            asv_args = list(asv_args) + ["-b", bench_filter]

    if _asv_builds_declared_environment(asv_args):
        _check_asv_python_version()

    site_path = spin.cmds.meson._get_site_packages(build_dir)
    if site_path is None:
        print("No built scikit-image found; run `spin build` first.")
        sys.exit(1)

    os.environ['PYTHONPATH'] = f'{site_path}{os.sep}:{os.environ.get("PYTHONPATH", "")}'
    spin.util.run(['asv'] + list(asv_args))
