#!/usr/bin/env python3
"""Determine the baseline/contender commits to compare, and the asv run
parameters (comparison factor, process count, skip-slow, full-params,
an optional benchmark-module filter, and the Python version) for the
current trigger event.

Required env vars:
  EVENT_NAME:        github.event_name
  GITHUB_SHA:        the current commit (provided by default by Actions)
  GITHUB_REPOSITORY: "owner/repo" (provided by default by Actions; used
                      for the nightly-baseline gh run list lookup)
  GH_TOKEN:          token for gh CLI (PR label lookup, nightly baseline
                      lookup - needs `actions: read` for the latter)
  PR_NUMBER:         pull request number (empty for non-PR events)
  PR_BASE_SHA:       github.event.pull_request.base.sha
  PR_BASE_LABEL:     github.event.pull_request.base.label
  PR_HEAD_SHA:       github.event.pull_request.head.sha
  PR_HEAD_LABEL:     github.event.pull_request.head.label
  DISPATCH_BASELINE: github.event.inputs.baseline (workflow_dispatch
                      only; "parent-commit" or "previous-nightly")

Writes should_run, baseline_sha, baseline_label, contender_label,
asv_factor, asv_processes, asv_skip_slow, asv_full_params,
bench_filter, and python_version to $GITHUB_OUTPUT.
"""

import json
import os
import subprocess

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
    event_name = os.environ["EVENT_NAME"]
    github_sha = os.environ["GITHUB_SHA"]

    if event_name == "pull_request":
        pr_base_sha = os.environ["PR_BASE_SHA"]
        pr_head_sha = os.environ["PR_HEAD_SHA"]
        baseline_sha = pr_base_sha
        baseline_label = os.environ["PR_BASE_LABEL"]
        contender_label = os.environ["PR_HEAD_LABEL"]
        asv_factor = "1.5"
        asv_processes = "2"
        asv_skip_slow = "1"
        asv_full_params = "0"

        # Fetch label state dynamically (not from the workflow trigger
        # payload) so that re-running the workflow after adding the
        # 'benchmark' label picks up the new label state correctly.
        labels = json.loads(
            run("gh", "pr", "view", os.environ["PR_NUMBER"], "--json", "labels")
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
        asv_factor = "1.5"
        asv_processes = "2"
        asv_skip_slow = "0"
        asv_full_params = "1"
        should_run = "true"
        bench_filter = ""

    else:
        # workflow_dispatch: compare against either the immediate
        # parent commit (default) or the commit compared by the last
        # successful nightly run, per the "baseline" dispatch input.
        # Falls back to the immediate parent if "previous-nightly" is
        # requested but no prior successful nightly run exists yet.
        dispatch_baseline = os.environ.get("DISPATCH_BASELINE") or "parent-commit"
        if dispatch_baseline == "previous-nightly":
            baseline_sha = resolve_baseline(last_nightly_sha(), github_sha)
        else:
            baseline_sha = run("git", "rev-parse", f"{github_sha}~1")
        baseline_label = baseline_sha
        contender_label = github_sha
        asv_factor = "1.5"
        asv_processes = "2"
        asv_skip_slow = "1"
        asv_full_params = "0"
        should_run = "true"
        bench_filter = ""

    # Single source of truth for the Python version used to build and
    # run the benchmarks, so it can't drift out of sync with asv's own
    # config.
    with open("asv.conf.json") as f:
        python_version = json.load(f)["pythons"][0]

    outputs = {
        "should_run": should_run,
        "baseline_sha": baseline_sha,
        "baseline_label": baseline_label,
        "contender_label": contender_label,
        "asv_factor": asv_factor,
        "asv_processes": asv_processes,
        "asv_skip_slow": asv_skip_slow,
        "asv_full_params": asv_full_params,
        "bench_filter": bench_filter,
        "python_version": python_version,
    }
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        for key, value in outputs.items():
            f.write(f"{key}={value}\n")


if __name__ == "__main__":
    main()
