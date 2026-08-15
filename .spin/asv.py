"""Benchmark commands for `spin`.

`spin asv` runs asv locally, and `spin asv --resolve` is what the
benchmark workflow calls to decide what a run compares. Both go through
the same definitions below, so a local run can reproduce what CI
measures rather than the two drifting apart.

benchmarks/README_CI.md describes the workflow around this.
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass

import click
import spin

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASV_CONF_FILE = os.path.join(_REPO_ROOT, "asv.conf.json")

# Written by --resolve for the benchmark job to pick up.
PARAMS_FILE = "benchmark-params.json"

# Where module names below live, and so the only paths a change has to
# touch for a benchmark to be worth running.
SOURCE_DIR = "src/skimage"

# Which benchmark modules cover each SOURCE_DIR subpackage. Subpackages
# absent here (color, data, draw, future, io) have no benchmarks, and
# benchmark_import_time covers the whole package rather than one
# subpackage, so neither scopes a run.
MODULE_MAP = {
    "exposure": ["benchmark_exposure"],
    "feature": ["benchmark_feature", "benchmark_peak_local_max"],
    "filters": ["benchmark_filters", "benchmark_rank"],
    "graph": ["benchmark_graph"],
    "measure": ["benchmark_measure"],
    "metrics": ["benchmark_metrics"],
    "morphology": ["benchmark_morphology"],
    "registration": ["benchmark_registration"],
    "restoration": ["benchmark_restoration"],
    "segmentation": ["benchmark_segmentation"],
    "transform": [
        "benchmark_transform",
        "benchmark_transform_warp",
        "benchmark_interpolation",
    ],
    "util": ["benchmark_util"],
}

# Trimmed run, for pull requests and manual dispatches: slow benchmarks
# skipped, reduced parameter matrices, and a comparison factor loose
# enough to absorb the noise of a shared runner.
FAST_PROFILE = {
    "ASV_FACTOR": "1.5",
    "ASV_PROCESSES": "2",
    "ASV_SKIP_SLOW": "1",
    "ASV_FULL_PARAMS": "0",
}

# Complete run, for the nightly: slow benchmarks and full parameter
# matrices included.
FULL_PROFILE = {**FAST_PROFILE, "ASV_SKIP_SLOW": "0", "ASV_FULL_PARAMS": "1"}

PROFILES = {"fast": FAST_PROFILE, "full": FULL_PROFILE}


def _git(*args, check=True, quiet=False):
    """Stdout of a git command, run from the repository root."""
    result = subprocess.run(
        ["git", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL if quiet else None,
        text=True,
        check=check,
        cwd=_REPO_ROOT,
    )
    return result.stdout.strip()


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def asv_pythons():
    """The Python versions asv.conf.json declares support for."""
    return _read_json(ASV_CONF_FILE)["pythons"]


def bench_filter_for_changes(base, head=""):
    """An asv -b regex covering the subpackages changed since base, or
    empty if none were.

    With no head, compares against the working tree so uncommitted
    edits count, which is what a local run wants. CI passes both
    commits. Paths under a subpackage MODULE_MAP doesn't list, or
    outside SOURCE_DIR, contribute nothing: they neither force nor
    block a run.
    """
    revisions = [base, head] if head else [base]
    changed = _git("diff", "--name-only", *revisions, "--", SOURCE_DIR).splitlines()

    modules = []
    for pkg, pkg_modules in MODULE_MAP.items():
        if any(path.startswith(f"{SOURCE_DIR}/{pkg}/") for path in changed):
            modules.extend(pkg_modules)

    return f"^({'|'.join(modules)})\\." if modules else ""


def default_base_ref():
    """The ref a local run compares against.

    The canonical repository's main branch, which CONTRIBUTING has
    contributors add as `upstream`, then a fork's, then a local one, so
    that none of them has to be spelled out. Deliberately not the
    branch's own tracking ref: for a pushed branch that is the same
    commit, leaving nothing to compare.
    """
    for ref in ("upstream/main", "origin/main", "main"):
        if _git("rev-parse", "--verify", ref, check=False, quiet=True):
            return ref
    return "main"


# --- Resolving what CI compares -------------------------------------


@dataclass
class Resolution:
    """What one trigger event resolved to."""

    baseline_sha: str
    baseline_label: str
    contender_label: str
    profile: dict
    should_run: bool = True
    # Empty runs every benchmark; otherwise an asv -b regex.
    bench_filter: str = ""

    @classmethod
    def between_shas(cls, baseline_sha, contender_sha, profile):
        """A resolution labelled by SHA, for the events that run off a
        branch and so have no base/head labels to show instead.
        """
        return cls(baseline_sha, baseline_sha, contender_sha, profile)


def _event_payload():
    """The github.event payload for this run."""
    path = os.environ.get("GITHUB_EVENT_PATH")
    return _read_json(path) if path and os.path.exists(path) else {}


def _gh(*args):
    """Stdout of a gh command that must succeed."""
    result = subprocess.run(
        ["gh", *args], stdout=subprocess.PIPE, text=True, check=True
    )
    return result.stdout.strip()


def last_nightly_sha():
    """The commit the last successful nightly compared, or empty."""
    output = _gh(
        "run", "list",
        "--repo", os.environ["GITHUB_REPOSITORY"],
        "--workflow=benchmarks.yaml", "--event=schedule",
        "--status=success", "--limit=1", "--json", "headSha"
    )  # fmt: skip
    runs = json.loads(output)
    return runs[0]["headSha"] if runs else ""


def baseline_or_parent(candidate, github_sha):
    """candidate, falling back to the parent commit when it's missing
    locally - no nightly has succeeded yet, say.
    """
    if candidate:
        found = subprocess.run(
            ["git", "cat-file", "-e", f"{candidate}^{{commit}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=_REPO_ROOT,
        )
        if found.returncode == 0:
            return candidate
    return _git("rev-parse", f"{github_sha}~1")


def has_benchmark_label(pr_number):
    """Whether the PR asks for the full suite by label.

    Read live rather than from the trigger payload, so re-running after
    adding the label picks up the new state.
    """
    labels = json.loads(_gh("pr", "view", str(pr_number), "--json", "labels"))
    return any("benchmark" in label["name"].lower() for label in labels["labels"])


def resolve_pull_request(event, github_sha):
    """Head against base, scoped to the subpackages touched.

    A benchmark label runs everything instead, since a change to shared
    code can move benchmarks outside the subpackage it lives in.
    """
    pull_request = event["pull_request"]
    base_sha = pull_request["base"]["sha"]
    resolution = Resolution(
        baseline_sha=base_sha,
        baseline_label=pull_request["base"]["label"],
        contender_label=pull_request["head"]["label"],
        profile=FAST_PROFILE,
    )
    if has_benchmark_label(pull_request["number"]):
        return resolution

    resolution.bench_filter = bench_filter_for_changes(
        base_sha, pull_request["head"]["sha"]
    )
    # Nothing benchmarked changed, so there's nothing worth running.
    resolution.should_run = bool(resolution.bench_filter)
    return resolution


def resolve_schedule(event, github_sha):
    """Main's tip against the commit the last nightly compared.

    Comparing against that rather than the immediate parent covers a
    busy day of merges cumulatively, not just its last commit.
    """
    baseline_sha = baseline_or_parent(last_nightly_sha(), github_sha)
    return Resolution.between_shas(baseline_sha, github_sha, FULL_PROFILE)


def resolve_workflow_dispatch(event, github_sha):
    """The parent commit, or the last nightly's, per the dispatch input."""
    if event.get("inputs", {}).get("baseline") == "previous-nightly":
        baseline_sha = baseline_or_parent(last_nightly_sha(), github_sha)
    else:
        baseline_sha = _git("rev-parse", f"{github_sha}~1")
    return Resolution.between_shas(baseline_sha, github_sha, FAST_PROFILE)


RESOLVERS = {
    "pull_request": resolve_pull_request,
    "schedule": resolve_schedule,
    "workflow_dispatch": resolve_workflow_dispatch,
}


def _write_outputs(resolution):
    """Publish what the build jobs need to start."""
    outputs = {
        "should_run": "true" if resolution.should_run else "false",
        # Taking the version from asv's own config keeps the two in step.
        "python_version": asv_pythons()[0],
        "baseline_sha": resolution.baseline_sha,
    }
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.writelines(f"{key}={value}\n" for key, value in outputs.items())


def _write_params(resolution):
    """Hand the benchmark job its asv settings, keyed by the variable
    name each is exported as so unpacking is a straight copy.
    """
    params = {
        **resolution.profile,
        "BASELINE_SHA": resolution.baseline_sha,
        "BASELINE_LABEL": resolution.baseline_label,
        "CONTENDER_LABEL": resolution.contender_label,
        "BENCH_FILTER": resolution.bench_filter,
    }
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)
        f.write("\n")


def resolve_for_ci():
    """Resolve the run and publish it for the rest of the workflow.

    Three values go to $GITHUB_OUTPUT for the jobs that start before
    benchmarking; the asv settings go to PARAMS_FILE, which the
    benchmark job downloads and unpacks into its environment.
    """
    resolve = RESOLVERS[os.environ["GITHUB_EVENT_NAME"]]
    resolution = resolve(_event_payload(), os.environ["GITHUB_SHA"])
    _write_outputs(resolution)
    _write_params(resolution)


# --- Running asv locally --------------------------------------------

# asv subcommands that build an environment from asv.conf.json. The
# rest either read stored results (compare, show) or don't touch
# environments at all (machine, publish, preview, ...).
_ASV_ENV_BUILDING = {"check", "continuous", "find", "profile", "run", "setup"}


def _asv_builds_declared_environment(asv_args):
    """Whether this asv invocation builds an environment matching
    asv.conf.json's declared `pythons`.

    It doesn't when the subcommand never builds one, or when
    `-E existing`/`--python=same` points asv at the interpreter already
    running, which works with any version.
    """
    if not asv_args or asv_args[0] not in _ASV_ENV_BUILDING:
        return False
    return not any(
        "existing" in arg or arg == "same" or arg.endswith((":same", "=same"))
        for arg in asv_args
    )


def _check_asv_python_version():
    """Fail fast if the active Python isn't one asv.conf.json declares,
    which asv would otherwise report from deep inside env creation.
    """
    pythons = asv_pythons()
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    if current not in pythons:
        print(
            f"Active Python is {current}, but asv.conf.json only declares "
            f"support for {pythons}. Rebuild with one of those versions "
            "before running `spin asv`."
        )
        sys.exit(1)


def _apply_asv_profile(name, asv_args):
    """Set up asv to measure what CI's named profile measures.

    ASV_SKIP_SLOW and ASV_FULL_PARAMS are read by benchmarks/__init__.py
    and go into the environment. The comparison factor and process count
    are asv's own flags, so they're only added where asv accepts them,
    and never over an equivalent flag already given explicitly.
    """
    profile = PROFILES[name]
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


@click.command()
@click.argument("asv_args", nargs=-1)
@click.option(
    "--resolve",
    is_flag=True,
    help="Resolve what the benchmark workflow should compare and write it "
    "to $GITHUB_OUTPUT and benchmark-params.json. For CI; runs nothing.",
)
@click.option(
    "--profile",
    type=click.Choice(sorted(PROFILES)),
    help="Measure what a 'fast' (trimmed, as pull request checks run) or "
    "'full' (nightly) run measures.",
)
@click.option(
    "--changed",
    is_flag=True,
    help="Only run the benchmarks covering subpackages you've changed, "
    "the way a pull request check scopes itself.",
)
@click.option(
    "--changed-base",
    default=None,
    help="Ref that --changed compares against  [default: upstream/main, "
    "else origin/main, else main]",
)
@spin.cmds.meson.build_dir_option
def asv(asv_args, resolve, profile, changed, changed_base, build_dir):
    """🏃 Run `asv` to collect benchmarks

    ASV_ARGS are passed through directly to asv, e.g.:

    spin asv -- check -v -E existing

    Pass --profile to match a CI run's benchmark selection and
    comparison factor, and --changed to scope the run to the
    subpackages you've touched, as a pull request check does:

    spin asv --profile fast --changed -- continuous main

    Please see CONTRIBUTING.txt
    """
    if resolve:
        resolve_for_ci()
        return

    if profile:
        asv_args = _apply_asv_profile(profile, asv_args)

    if changed:
        if any(a == "-b" or a.startswith("--bench") for a in asv_args):
            print("--changed was ignored: an explicit -b/--bench filter is set.")
        else:
            base = changed_base or default_base_ref()
            merge_base = _git("merge-base", base, "HEAD")
            bench_filter = bench_filter_for_changes(merge_base)
            if not bench_filter:
                print(
                    f"No benchmarked subpackage changed since {base}; nothing to run."
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
