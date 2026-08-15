#!/usr/bin/env python3
"""Decide what the benchmark workflow should compare, and how.

Each trigger event gets a resolver below that returns a Resolution:
which commits to compare, whether to run at all, which benchmarks to
run, and which profile's asv settings to use. The asv settings
themselves come from benchmarks/profiles.json and the subpackage
scoping from benchmarks/module-map.json, both of which `spin asv`
reads too, so a local run can reproduce what CI measures.

Reads the environment Actions already provides - GITHUB_EVENT_NAME,
GITHUB_SHA, GITHUB_REPOSITORY, GITHUB_EVENT_PATH (the full event
payload, which carries the pull_request and workflow_dispatch input
fields), GITHUB_OUTPUT - plus GH_TOKEN for the gh CLI lookups (PR label
lookup, nightly baseline lookup; the latter needs `actions: read`).

Writes should_run, python_version, and baseline_sha to $GITHUB_OUTPUT:
the only values other jobs need in order to start. The rest of the asv
parameters go to benchmark-params.json, which the benchmark job
downloads and unpacks into $GITHUB_ENV (see prepare-benchmarks.sh), so
they don't have to be threaded through job outputs one `env:` entry at
a time.
"""

import json
import os
import subprocess
from dataclasses import dataclass

PARAMS_FILE = "benchmark-params.json"
PROFILES_FILE = "benchmarks/profiles.json"
MODULE_MAP_FILE = "benchmarks/module-map.json"
ASV_CONF_FILE = "asv.conf.json"


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


def run(*args: str) -> str:
    return subprocess.run(
        args, stdout=subprocess.PIPE, text=True, check=True
    ).stdout.strip()


def event_payload() -> dict:
    """The github.event payload for this run."""
    path = os.environ.get("GITHUB_EVENT_PATH")
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_profile(name: str) -> dict:
    """The asv settings for a named run profile."""
    with open(PROFILES_FILE) as f:
        profile = json.load(f)[name]
    return {k: v for k, v in profile.items() if k != "description"}


def commit_exists(sha: str) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def last_nightly_sha() -> str:
    """The commit compared by the last successful nightly (schedule)
    run of this workflow, or empty if none exists yet.
    """
    output = run(
        "gh",
        "run",
        "list",
        "--repo",
        os.environ["GITHUB_REPOSITORY"],
        "--workflow=benchmarks.yaml",
        "--event=schedule",
        "--status=success",
        "--limit=1",
        "--json",
        "headSha",
    )
    runs = json.loads(output)
    return runs[0]["headSha"] if runs else ""


def baseline_or_parent(candidate: str, github_sha: str) -> str:
    """candidate, or the immediate parent commit if candidate is empty
    or isn't available locally (e.g. no nightly run has succeeded yet).
    """
    if candidate and commit_exists(candidate):
        return candidate
    return run("git", "rev-parse", f"{github_sha}~1")


def has_benchmark_label(pr_number: int) -> bool:
    """Whether the PR carries a label asking for the full suite.

    Read live rather than from the trigger payload, so re-running the
    workflow after adding the label picks up the new state.
    """
    labels = json.loads(run("gh", "pr", "view", str(pr_number), "--json", "labels"))
    return any("benchmark" in label["name"].lower() for label in labels["labels"])


def bench_filter_for_changes(base_sha: str, head_sha: str) -> str:
    """An asv -b regex covering the subpackages changed between two
    commits, per benchmarks/module-map.json, or "" if none were.

    Subpackages absent from the map, or changes outside src/skimage/
    entirely (docs, CI config, etc.), contribute no module: they
    neither force nor block a run. `spin asv --changed` scopes local
    runs from the same map (see .spin/cmds.py).
    """
    changed = run(
        "git", "diff", "--name-only", base_sha, head_sha, "--", "src/skimage/"
    ).splitlines()
    with open(MODULE_MAP_FILE) as f:
        module_map = json.load(f)

    modules = []
    for pkg, pkg_modules in module_map.items():
        if pkg.startswith("_"):
            continue
        if any(line.startswith(f"src/skimage/{pkg}/") for line in changed):
            modules.extend(pkg_modules)

    return f"^({'|'.join(modules)})\\." if modules else ""


def resolve_pull_request(event: dict, github_sha: str) -> Resolution:
    """Compare the PR head against its base, running only the
    benchmarks covering the subpackages it touches - or the whole suite
    if it carries a benchmark label, since a change to shared code can
    move benchmarks outside the subpackage it lives in.
    """
    pull_request = event["pull_request"]
    base_sha = pull_request["base"]["sha"]
    resolution = Resolution(
        baseline_sha=base_sha,
        baseline_label=pull_request["base"]["label"],
        contender_label=pull_request["head"]["label"],
        profile=load_profile("fast"),
    )

    if has_benchmark_label(pull_request["number"]):
        return resolution

    resolution.bench_filter = bench_filter_for_changes(
        base_sha, pull_request["head"]["sha"]
    )
    # Nothing benchmarked changed, so there is nothing worth running.
    resolution.should_run = bool(resolution.bench_filter)
    return resolution


def resolve_schedule(event: dict, github_sha: str) -> Resolution:
    """Compare main's tip against the commit the last successful
    nightly compared, rather than its immediate parent, so a busy day
    of merges is covered cumulatively rather than only its most recent
    commit. Runs the full suite, untrimmed and unscoped.
    """
    baseline_sha = baseline_or_parent(last_nightly_sha(), github_sha)
    return Resolution(
        baseline_sha=baseline_sha,
        baseline_label=baseline_sha,
        contender_label=github_sha,
        profile=load_profile("full"),
    )


def resolve_workflow_dispatch(event: dict, github_sha: str) -> Resolution:
    """Compare against either the immediate parent commit or the
    commit the last successful nightly compared, per the "baseline"
    dispatch input.
    """
    if event.get("inputs", {}).get("baseline") == "previous-nightly":
        baseline_sha = baseline_or_parent(last_nightly_sha(), github_sha)
    else:
        baseline_sha = run("git", "rev-parse", f"{github_sha}~1")
    return Resolution(
        baseline_sha=baseline_sha,
        baseline_label=baseline_sha,
        contender_label=github_sha,
        profile=load_profile("fast"),
    )


RESOLVERS = {
    "pull_request": resolve_pull_request,
    "schedule": resolve_schedule,
    "workflow_dispatch": resolve_workflow_dispatch,
}


def write_outputs(resolution: Resolution) -> None:
    """Publish what other jobs need before the benchmark step itself:
    which Python to build against, what to build as the baseline, and
    whether to bother at all.
    """
    # Single source of truth for the Python version used to build and
    # run the benchmarks, so it can't drift out of sync with asv.
    with open(ASV_CONF_FILE) as f:
        python_version = json.load(f)["pythons"][0]

    outputs = {
        "should_run": "true" if resolution.should_run else "false",
        "python_version": python_version,
        "baseline_sha": resolution.baseline_sha,
    }
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        for key, value in outputs.items():
            f.write(f"{key}={value}\n")


def write_params(resolution: Resolution) -> None:
    """Hand the benchmark job its asv settings, keyed by the
    environment variable name each is exported as, so unpacking is a
    straight copy into $GITHUB_ENV.
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


def main() -> None:
    event_name = os.environ["GITHUB_EVENT_NAME"]
    resolve = RESOLVERS[event_name]
    resolution = resolve(event_payload(), os.environ["GITHUB_SHA"])
    write_outputs(resolution)
    write_params(resolution)


if __name__ == "__main__":
    main()
