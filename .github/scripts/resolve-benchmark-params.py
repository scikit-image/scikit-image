#!/usr/bin/env python3
"""Determine the baseline/contender commits to compare, and the asv run
parameters (comparison factor, process count, skip-slow, full-params,
an optional benchmark-module filter, and the Python version) for the
current trigger event.

The asv settings themselves come from benchmarks/profiles.json, which
`spin asv --profile` reads too, so a local run can reproduce what CI
measures.

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

PARAMS_FILE = "benchmark-params.json"
PROFILES_FILE = "benchmarks/profiles.json"

# Subpackages with no corresponding benchmark file, or changes outside
# src/skimage/ entirely (docs, CI config, etc.), simply don't
# contribute a module - they neither force nor block a run.
PACKAGES = [
    "exposure",
    "feature",
    "filters",
    "graph",
    "measure",
    "metrics",
    "morphology",
    "registration",
    "restoration",
    "segmentation",
    "transform",
    "util",
]

# Packages whose benchmark coverage spans more than one module file.
EXTRA_MODULES = {
    "feature": "benchmark_feature|benchmark_peak_local_max",
    "filters": "benchmark_filters|benchmark_rank",
    "transform": "benchmark_transform|benchmark_transform_warp|benchmark_interpolation",
}


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


def bench_modules_for_path_changes(changed: str) -> str:
    """Map changed src/skimage/<subpackage>/ paths to the benchmark
    module(s) that cover them.
    """
    changed_lines = changed.splitlines()
    modules = []
    for pkg in PACKAGES:
        if not any(line.startswith(f"src/skimage/{pkg}/") for line in changed_lines):
            continue
        modules.append(EXTRA_MODULES.get(pkg, f"benchmark_{pkg}"))
    return "|".join(modules)


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


def load_profile(name: str) -> dict:
    """The asv settings for a named run profile.

    Shared with `spin asv --profile` (see .spin/cmds.py) so a local run
    can reproduce what CI measures instead of the two drifting apart.
    """
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


def resolve_baseline(candidate: str, github_sha: str) -> str:
    """candidate, or the immediate parent commit if candidate is empty
    or isn't available locally (e.g. no nightly run has succeeded
    yet).
    """
    if candidate and commit_exists(candidate):
        return candidate
    return run("git", "rev-parse", f"{github_sha}~1")


def main() -> None:
    event_name = os.environ["GITHUB_EVENT_NAME"]
    github_sha = os.environ["GITHUB_SHA"]
    event = event_payload()

    if event_name == "pull_request":
        pull_request = event["pull_request"]
        pr_base_sha = pull_request["base"]["sha"]
        pr_head_sha = pull_request["head"]["sha"]
        baseline_sha = pr_base_sha
        baseline_label = pull_request["base"]["label"]
        contender_label = pull_request["head"]["label"]
        profile = load_profile("pr")

        # Fetch label state dynamically (not from the workflow trigger
        # payload) so that re-running the workflow after adding the
        # 'benchmark' label picks up the new label state correctly.
        labels = json.loads(
            run("gh", "pr", "view", str(pull_request["number"]), "--json", "labels")
        )["labels"]
        has_benchmark_label = any(
            "benchmark" in label["name"].lower() for label in labels
        )

        if has_benchmark_label:
            # Explicit override: always run the full suite, regardless
            # of which paths changed.
            should_run = "true"
            bench_filter = ""
        else:
            changed = run(
                "git",
                "diff",
                "--name-only",
                pr_base_sha,
                pr_head_sha,
                "--",
                "src/skimage/",
            )
            modules = bench_modules_for_path_changes(changed)
            if modules:
                should_run = "true"
                bench_filter = f"^({modules})\\."
            else:
                should_run = "false"
                bench_filter = ""

    elif event_name == "schedule":
        # Nightly: full, untrimmed suite against the commit compared
        # by the last successful nightly run, not just main's
        # immediate parent - this catches cumulative drift across
        # however many PRs merged since then, not just the single
        # most recent one. Falls back to the immediate parent if no
        # prior successful nightly run exists yet (e.g. the very
        # first time this schedule fires).
        prev_sha = resolve_baseline(last_nightly_sha(), github_sha)
        baseline_sha = prev_sha
        baseline_label = prev_sha
        contender_label = github_sha
        profile = load_profile("nightly")
        should_run = "true"
        bench_filter = ""

    else:
        # workflow_dispatch: compare against either the immediate
        # parent commit (default) or the commit compared by the last
        # successful nightly run, per the "baseline" dispatch input.
        # Falls back to the immediate parent if "previous-nightly" is
        # requested but no prior successful nightly run exists yet.
        dispatch_baseline = event.get("inputs", {}).get("baseline") or "parent-commit"
        if dispatch_baseline == "previous-nightly":
            baseline_sha = resolve_baseline(last_nightly_sha(), github_sha)
        else:
            baseline_sha = run("git", "rev-parse", f"{github_sha}~1")
        baseline_label = baseline_sha
        contender_label = github_sha
        profile = load_profile("pr")
        should_run = "true"
        bench_filter = ""

    # Single source of truth for the Python version used to build and
    # run the benchmarks, so it can't drift out of sync with asv's own
    # config.
    with open("asv.conf.json") as f:
        python_version = json.load(f)["pythons"][0]

    # Only what other jobs need before the benchmark step itself:
    # which Python to build against, what to build as the baseline,
    # and whether to bother at all.
    outputs = {
        "should_run": should_run,
        "python_version": python_version,
        "baseline_sha": baseline_sha,
    }
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        for key, value in outputs.items():
            f.write(f"{key}={value}\n")

    # Keyed by the environment variable name each value is exported as,
    # so unpacking is a straight copy into $GITHUB_ENV.
    params = {
        **profile,
        "BASELINE_SHA": baseline_sha,
        "BASELINE_LABEL": baseline_label,
        "CONTENDER_LABEL": contender_label,
        "BENCH_FILTER": bench_filter,
    }
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
