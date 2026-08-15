#!/usr/bin/env python3
"""Resolve what the benchmark workflow compares, and how.

One resolver per trigger event returns a Resolution, which main()
writes out: three values to $GITHUB_OUTPUT for the jobs that run
before benchmarking, the rest to benchmark-params.json for the
benchmark job itself. Settings come from benchmarks/profiles.json and
benchmarks/module-map.json, which `spin asv` reads too so local runs
can match CI. benchmarks/README_CI.md describes the whole flow.

Expects GITHUB_EVENT_NAME, GITHUB_SHA, GITHUB_REPOSITORY,
GITHUB_EVENT_PATH, GITHUB_OUTPUT, and GH_TOKEN for the gh lookups (the
nightly baseline one needs `actions: read`).
"""

import importlib.util
import json
import os
import subprocess
from dataclasses import dataclass

PARAMS_FILE = "benchmark-params.json"

# Loaded by path because benchmarks/__init__.py imports numpy and
# skimage, which this job has no reason to install.
_spec = importlib.util.spec_from_file_location("config", "benchmarks/config.py")
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)


def run(*args: str) -> str:
    """Stdout of a command that must succeed.

    Stderr is left alone so git and gh report their own failures into
    the job log.
    """
    result = subprocess.run(args, stdout=subprocess.PIPE, text=True, check=True)
    return result.stdout.strip()


def read_json(path: str):
    with open(path) as f:
        return json.load(f)


def event_payload() -> dict:
    """The github.event payload for this run."""
    path = os.environ.get("GITHUB_EVENT_PATH")
    return read_json(path) if path and os.path.exists(path) else {}


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
    def between_shas(cls, baseline_sha: str, contender_sha: str, profile: dict):
        """A resolution labelled by SHA, for the events that run off a
        branch and so have no base/head labels to show instead.
        """
        return cls(baseline_sha, baseline_sha, contender_sha, profile)


def last_nightly_sha() -> str:
    """The commit the last successful nightly compared, or empty."""
    output = run(
        "gh", "run", "list",
        "--repo", os.environ["GITHUB_REPOSITORY"],
        "--workflow=benchmarks.yaml", "--event=schedule",
        "--status=success", "--limit=1", "--json", "headSha"
    )  # fmt: skip
    runs = json.loads(output)
    return runs[0]["headSha"] if runs else ""


def baseline_or_parent(candidate: str, github_sha: str) -> str:
    """candidate, falling back to the parent commit when it's missing
    locally - no nightly has succeeded yet, say.
    """
    if candidate:
        found = subprocess.run(
            ["git", "cat-file", "-e", f"{candidate}^{{commit}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if found.returncode == 0:
            return candidate
    return run("git", "rev-parse", f"{github_sha}~1")


def has_benchmark_label(pr_number: int) -> bool:
    """Whether the PR asks for the full suite by label.

    Read live rather than from the trigger payload, so re-running after
    adding the label picks up the new state.
    """
    labels = json.loads(run("gh", "pr", "view", str(pr_number), "--json", "labels"))
    return any("benchmark" in label["name"].lower() for label in labels["labels"])


def bench_filter_for_changes(base_sha: str, head_sha: str) -> str:
    """An asv -b regex covering the subpackages changed between two
    commits, or empty if none were.
    """
    changed = run(
        "git", "diff", "--name-only", base_sha, head_sha, "--", "src/skimage/"
    ).splitlines()
    return config.bench_filter(changed)


def resolve_pull_request(event: dict, github_sha: str) -> Resolution:
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
        profile=config.load_profile("fast"),
    )
    if has_benchmark_label(pull_request["number"]):
        return resolution

    resolution.bench_filter = bench_filter_for_changes(
        base_sha, pull_request["head"]["sha"]
    )
    # Nothing benchmarked changed, so there's nothing worth running.
    resolution.should_run = bool(resolution.bench_filter)
    return resolution


def resolve_schedule(event: dict, github_sha: str) -> Resolution:
    """Main's tip against the commit the last nightly compared.

    Comparing against that rather than the immediate parent covers a
    busy day of merges cumulatively, not just its last commit.
    """
    baseline_sha = baseline_or_parent(last_nightly_sha(), github_sha)
    return Resolution.between_shas(
        baseline_sha, github_sha, config.load_profile("full")
    )


def resolve_workflow_dispatch(event: dict, github_sha: str) -> Resolution:
    """The parent commit, or the last nightly's, per the dispatch input."""
    if event.get("inputs", {}).get("baseline") == "previous-nightly":
        baseline_sha = baseline_or_parent(last_nightly_sha(), github_sha)
    else:
        baseline_sha = run("git", "rev-parse", f"{github_sha}~1")
    return Resolution.between_shas(
        baseline_sha, github_sha, config.load_profile("fast")
    )


RESOLVERS = {
    "pull_request": resolve_pull_request,
    "schedule": resolve_schedule,
    "workflow_dispatch": resolve_workflow_dispatch,
}


def write_outputs(resolution: Resolution) -> None:
    """Publish what the build jobs need to start."""
    outputs = {
        "should_run": "true" if resolution.should_run else "false",
        # Taking the version from asv's own config keeps the two in step.
        "python_version": config.asv_pythons()[0],
        "baseline_sha": resolution.baseline_sha,
    }
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.writelines(f"{key}={value}\n" for key, value in outputs.items())


def write_params(resolution: Resolution) -> None:
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


def main() -> None:
    resolve = RESOLVERS[os.environ["GITHUB_EVENT_NAME"]]
    resolution = resolve(event_payload(), os.environ["GITHUB_SHA"])
    write_outputs(resolution)
    write_params(resolution)


if __name__ == "__main__":
    main()
